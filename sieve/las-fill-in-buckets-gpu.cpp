#include "cado.h" // IWYU pragma: keep

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>

#include "bucket.hpp"
#include "bucket-push-update.hpp" // IWYU pragma: keep
#include "fb.hpp"
#include "fb-types.hpp"
#include "las-bkmult.hpp"
#include "las-config.hpp"
#include "las-fbroot-qlattice.hpp"
#include "las-fill-in-buckets-gpu.hpp"
#include "las-plattice.hpp"
#include "las-qlattice.hpp"
#include "las-threads-work-data.hpp"
#include "las-auxiliary-data.hpp"
#include "las-where-am-i.hpp"
#include "macros.h"
#include "tdict.hpp"
#include "threadpool.hpp"

#include "gpu/las-gpu-engine.hpp"
#include "gpu/las-gpu-fib-plan.hpp"
#include "gpu/las-gpu-select.hpp"

/* The bucket update that the device produces must be the one the rest
 * of the sieve reads. bucket_update_t gets its layout from inheriting
 * shorthint_t before bare_bucket_update_t, which is not something the
 * device header can know; pin it here instead.
 */
using fib_update_t = bucket_update_t<1, shorthint_t>;
static_assert(sizeof(fib_update_t) == sizeof(gpu_fib_update));
static_assert(offsetof(gpu_fib_update, hint) == 0);
static_assert(offsetof(gpu_fib_update, x) == sizeof(uint16_t));
static_assert(std::is_same_v<decltype(fib_update_t::hint), uint16_t>);
static_assert(std::is_same_v<decltype(fib_update_t::x), uint16_t>);

/* {{{ the cache */

/* The flattened form of one factor base slicing, and its device copy. */
struct las_gpu_fib_image {
    gpu_engine * eng = nullptr;
    gpu_fib_task * task = nullptr;

    std::vector<gpu_fib_entry> entries;
    std::vector<gpu_fib_slice> slices;
    std::vector<slice_index_t> slice_indices;
    std::vector<double> weights;


    las_gpu_fib_image() = default;
    las_gpu_fib_image(las_gpu_fib_image const &) = delete;
    las_gpu_fib_image & operator=(las_gpu_fib_image const &) = delete;

    ~las_gpu_fib_image()
    {
        if (eng && task)
            eng->fib_task_destroy(task);
    }

    /* which slice a flat entry index belongs to */
    size_t slice_of(uint32_t k) const
    {
        auto it = std::upper_bound(slices.begin(), slices.end(), k,
                [](uint32_t v, gpu_fib_slice const & s) {
                    return v < s.first;
                });
        ASSERT_ALWAYS(it != slices.begin());
        return size_t((it - 1) - slices.begin());
    }
};

struct las_gpu_fib_cache {
    /* One image per factor base slicing. A side does not have just one:
     * the slicing is keyed on, among other things, the lognorm scale,
     * which depends on the special-q, so las goes back and forth
     * between a handful of them. They are cached by las_info and live
     * as long as it does, so their addresses are stable keys. Keeping
     * only the latest image here would rebuild and re-upload the
     * factor base on most special-q's. */
    gpu_engine * eng = nullptr;
    std::map<void const *, std::unique_ptr<las_gpu_fib_image>> images;
};

void las_gpu_fib_cache_deleter::operator()(las_gpu_fib_cache * p) const
{
    delete p;
}

/* }}} */
/* {{{ building the flattened factor base */

static std::unique_ptr<las_gpu_fib_image>
build_image(fb_factorbase::slicing const & fbs)
{
    auto I = std::make_unique<las_gpu_fib_image>();
    fb_factorbase::slicing::part const & P = fbs.get_part(1);

    P.foreach_slice([&](auto const & sl) {
        using entry_t = typename std::decay_t<decltype(sl)>::entry_t;
        if constexpr (las_gpu_fib_takes_slice<entry_t>()) {
            gpu_fib_slice gs {};
            gs.first = uint32_t(I->entries.size());
            gs.slice_index = sl.get_index();
            slice_offset_t i_entry = 0;
            for (auto const & e: sl) {
                /* one lattice per root, which is the unit of work the
                 * kernel parallelises over */
                for (int k = 0; k < entry_t::fixed_nr_roots; k++)
                    I->entries.push_back(gpu_fib_entry {
                            e.p, e.invq, e.roots[size_t(k)], i_entry, 0});
                i_entry++;
            }
            gs.count = uint32_t(I->entries.size()) - gs.first;
            I->slices.push_back(gs);
            I->slice_indices.push_back(sl.get_index());
            I->weights.push_back(sl.get_weight());
        }
    });

    I->eng = las_gpu::engine();
    I->task = I->eng->fib_task_create(I->entries.data(),
                                      uint32_t(I->entries.size()),
                                      I->slices.data(),
                                      uint32_t(I->slices.size()));
    return I;
}

/* }}} */
/* {{{ what the device hands back */

/* Replays the ordinary code for one lattice the kernel declined. These
 * are projective-like, which is to say they contribute row updates --
 * a separate per-bucket structure carrying its own slice index -- so
 * adding them here does not disturb the slice boundaries the device
 * just laid out.
 */
static void replay_one(bucket_array_t<1, shorthint_t> & BA,
                       gpu_fib_entry const & e, slice_index_t slice_index,
                       qlattice_basis const & Q, int logI, uint32_t J,
                       where_am_I & w)
{
    fb_root_p1 const R = fb_root_in_qlattice(
            e.p, fb_root_p1::affine_root(e.r), e.invq, Q);
    plattice_info const pli(e.p, R.r, R.proj, logI);
    plattice_enumerator ple(pli, e.hint, logI);
    plattice_enumerator::fence const F(logI, int(J));

    ple.next(F);

    if (pli.is_discarded())
        return;

    /* Anything else would have gone into the per-slice update stream,
     * which is closed by now. The kernel and the code above agree on
     * the classification -- tests/sieve/test_fib_gpu is what says so --
     * so this is a consistency check, not a fallback.
     */
    ASSERT_ALWAYS(ple.is_projective_like(logI));

    int const logB = LOG_BUCKET_REGIONS[1];
    auto const bmask = (UINT64_C(1) << logB) - 1;
    uint32_t const maskB1I = (UINT32_C(1) << std::min(logB, logI)) - 1;

    fib_update_t u(0, 0, e.hint, slice_index);
    while (!ple.done(F)) {
        u.set_x(uint16_t(ple.get_x() & bmask));
        int const N = int(ple.get_x() >> logB);
        int const n = ple.advance_to_end_of_row_or_smallest_region(maskB1I);
        BA.push_row_update(slice_index, ple.get_inc_step(), N, n, u, w);
        ple.next(F);
    }
}

/* }}} */
/* {{{ cross-check against the host backend

   Set LAS_GPU_FIB_CROSSCHECK=1 to run the portable backend over the
   same input right after the real one and compare. It is far too slow
   for production -- the host backend is a reference implementation,
   not a fast one -- but it is the only way to catch a device-side
   disagreement on a real factor base, with real q-lattices and real
   bucket geometry, rather than on the synthetic input of
   tests/sieve/test_fib_engine.

   Updates are compared per (bucket, slice) as multisets: the device
   scatters with atomics, so it does not reproduce the order the host
   walks in, and nothing downstream depends on that order.
 */

static void crosscheck(las_gpu_fib_image const & I, gpu_fib_plan const & plan,
                       gpu_fib_output const & out, int side)
{
    /* one reference task per image, never freed: this is a debugging
     * aid, and the images themselves live until las exits */
    static gpu_engine * ref = gpu_engine_cpu_create();
    static std::mutex mu;
    static std::map<las_gpu_fib_image const *, gpu_fib_task *> reftasks;
    gpu_fib_task * reftask;
    {
        std::lock_guard<std::mutex> const lock(mu);
        gpu_fib_task *& t = reftasks[&I];
        if (!t)
            t = ref->fib_task_create(I.entries.data(),
                                     uint32_t(I.entries.size()),
                                     I.slices.data(),
                                     uint32_t(I.slices.size()));
        reftask = t;
    }

    gpu_fib_pass * const rp = ref->fib_launch(reftask, plan);
    gpu_fib_output const r = ref->fib_wait(rp);

    auto fail = [&](char const * what) {
        fprintf(stderr, "las-gpu crosscheck: side %d: %s\n", side, what);
        abort();
    };

    if (r.nslices != out.nslices || r.n_buckets != out.n_buckets)
        fail("different geometry");
    if (r.nhost != out.nhost)
        fail("different number of entries handed back");
    {
        std::vector<uint32_t> a(out.host_entries, out.host_entries + out.nhost);
        std::vector<uint32_t> b(r.host_entries, r.host_entries + r.nhost);
        std::ranges::sort(a);
        std::ranges::sort(b);
        if (a != b)
            fail("different entries handed back");
    }

    size_t const grid = size_t(out.nslices) * out.n_buckets;
    for (size_t i = 0; i < grid; i++)
        if (out.counts[i] != r.counts[i] || out.offsets[i] != r.offsets[i]) {
            fprintf(stderr,
                    "las-gpu crosscheck: side %d bucket %zu slice %zu:"
                    " device %u@%u, reference %u@%u\n",
                    side, i / out.nslices, i % out.nslices,
                    out.counts[i], out.offsets[i], r.counts[i], r.offsets[i]);
            abort();
        }

    auto key = [](gpu_fib_update const & u) {
        return (uint32_t(u.x) << 16) | u.hint;
    };
    for (size_t i = 0; i < grid; i++) {
        uint32_t const o = out.offsets[i];
        uint32_t const n = out.counts[i];
        std::vector<uint32_t> va(n), vb(n);
        for (uint32_t k = 0; k < n; k++) {
            va[k] = key(out.updates[o + k]);
            vb[k] = key(r.updates[o + k]);
        }
        std::ranges::sort(va);
        std::ranges::sort(vb);
        if (va != vb) {
            fprintf(stderr,
                    "las-gpu crosscheck: side %d bucket %zu slice %zu:"
                    " %u updates differ\n",
                    side, i / out.nslices, i % out.nslices, n);
            abort();
        }
    }

    ref->fib_release(rp);
}

/* }}} */
/* {{{ the pass */

bool las_gpu_fib_usable(nfs_work const & ws MAYBE_UNUSED,
                        qlattice_basis const & Q MAYBE_UNUSED)
{
#if defined(TRACE_K) || defined(SIQS_SIEVE) || defined(SUPPORT_LARGE_Q)
    /* The device path covers the plain case only. It cannot report
     * where_am_I hits; it implements the 31-bit root transform, not the
     * 127-bit one that a q above 2^31 needs; and SIQS fills its buckets
     * differently altogether. */
    return false;
#else
    if (!las_gpu::stage_enabled(las_gpu::STAGE_BUCKETS))
        return false;
    if (ws.toplevel != 1)
        return false;
    if (Q.sublat.m)
        return false;
    /* The kernel filters the primes dividing the special-q against a
     * small fixed-size list, like Q.is_coprime_to() does. A q with more
     * factors than that is not worth a special case. */
    if (Q.doing.prime_factors.size() > 8)
        return false;
    return true;
#endif
}

/* One device pass. It outlives the special-q's bucket filling: the
 * bucket arrays point into its output, so it is held by the side data
 * until the next special-q's pass replaces it. */
struct las_gpu_fib_inflight {
    las_gpu_fib_image & I;
    gpu_fib_plan plan;
    gpu_fib_pass * pass;
    int side;
    gpu_fib_output out {};

    las_gpu_fib_inflight(las_gpu_fib_image & I, gpu_fib_plan const & plan,
                         int side)
        : I(I), plan(plan), pass(I.eng->fib_launch(I.task, plan)), side(side)
    {}
    las_gpu_fib_inflight(las_gpu_fib_inflight const &) = delete;
    las_gpu_fib_inflight & operator=(las_gpu_fib_inflight const &) = delete;

    ~las_gpu_fib_inflight() { I.eng->fib_release(pass); }
};

/* Wait for the device, then hand its output to a bucket array -- any
 * one: apply_one_bucket and the purge walk every array of the side, and
 * the device's updates take no room in them. The lattices that the
 * device handed back are replayed into the same array. */
static void collect(worker_thread * worker, int side, nfs_work & ws,
                    nfs_aux & aux, qlattice_basis const & Q,
                    std::shared_ptr<las_gpu_fib_inflight> const & F)
{
    /* The where_am_I we work with is the worker's own, exactly as in
     * fill_in_buckets_toplevel_wrapper. */
    int const id = worker->rank();
    nfs_aux::thread_data & taux(aux.th[id]);
    timetree_t & timer(aux.get_timer(worker));
    where_am_I & w(taux.w);
    nfs_work::side_data & wss(ws.sides[side]);

    ENTER_THREAD_TIMER(timer);
    MARK_TIMER_FOR_SIDE(timer, side);
    CHILD_TIMER(timer, "fill_in_buckets_toplevel_gpu");

    WHERE_AM_I_UPDATE(w, side, side);
    WHERE_AM_I_UPDATE(w, N, 0);

    las_gpu_fib_image const & I(F->I);
    F->out = I.eng->fib_wait(F->pass);
    gpu_fib_output const & out(F->out);

    static bool const xcheck = getenv("LAS_GPU_FIB_CROSSCHECK") != nullptr;
    if (xcheck)
        crosscheck(I, F->plan, out, side);

    auto acquired = wss.reserve_BA<1, shorthint_t>();
    auto & BA = acquired.access();
    ASSERT_ALWAYS(BA.n_bucket == out.n_buckets);

    BA.add_external_slices(I.slice_indices.data(), 0, out.nslices,
            reinterpret_cast<fib_update_t const *>(out.updates),
            out.offsets, out.nslices);

    for (uint64_t k = 0; k < out.nhost; k++) {
        uint32_t const idx = out.host_entries[k];
        size_t const s = I.slice_of(idx);
        replay_one(BA, I.entries[idx], I.slices[s].slice_index, Q,
                   ws.conf.logI, ws.J, w);
    }
}

double las_gpu_fib_start(nfs_work & ws, nfs_aux & aux,
                         qlattice_basis const & Q, int side,
                         thread_pool & pool)
{
    nfs_work::side_data & wss(ws.sides[side]);

    /* The previous special-q is done with it, and the bucket arrays
     * that pointed into it have been reset. Give it back before taking
     * a new one, so that a side never holds two. */
    wss.gpu_fib_pass.reset();

    if (!wss.gpu_fib || wss.gpu_fib->eng != las_gpu::engine()) {
        wss.gpu_fib.reset(new las_gpu_fib_cache);
        wss.gpu_fib->eng = las_gpu::engine();
    }
    auto & slot = wss.gpu_fib->images[wss.fbs];
    if (!slot)
        slot = build_image(*wss.fbs);
    las_gpu_fib_image & I(*slot);

    if (I.slices.empty())
        return 0;

    uint32_t const nb = uint32_t(ws.nb_buckets[1]);

    /* The kernels index the per-bucket arrays by x >> log_bucket_region
     * for every x below J << logI, so the bucket arrays have to be at
     * least that wide. They always are -- las rounds the bucket count
     * up, and truncates J afterwards rather than before -- but an
     * overrun here would be a silent out-of-bounds atomic on the
     * device, so it is worth one comparison per special-q to be sure. */
    ASSERT_ALWAYS((uint64_t(ws.J) << ws.conf.logI) <=
                  (uint64_t(nb) << LOG_BUCKET_REGIONS[1]));

    gpu_fib_plan plan {};
    plan.logI = ws.conf.logI;
    plan.log_bucket_region = LOG_BUCKET_REGIONS[1];
    plan.J = ws.J;
    plan.n_buckets = nb;
    plan.basis = gpu_qlattice {Q.a0, Q.b0, Q.a1, Q.b1};

    /* Mirror of Q.is_coprime_to(): only the factors that a factor base
     * prime could possibly equal are worth listing. */
    plan.n_q_factors = 0;
    for (uint64_t f: Q.doing.prime_factors) {
        if (f <= UINT32_MAX)
            plan.q_factors[plan.n_q_factors++] = uint32_t(f);
    }

    auto F = std::make_shared<las_gpu_fib_inflight>(I, plan, side);
    wss.gpu_fib_pass = F;

    /* Cost 0, so that the pool gets to it last: by then the device has
     * had all the time the ordinary tasks took. */
    pool.add_task(thread_pool::QUEUE_GENERIC, 0, collect,
                  side, std::ref(ws), std::ref(aux), std::cref(Q), F);

    return std::accumulate(I.weights.begin(), I.weights.end(), 0.0);
}

/* }}} */
