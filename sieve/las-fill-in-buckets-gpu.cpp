#include "cado.h" // IWYU pragma: keep

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <memory>
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

struct las_gpu_fib_cache {
    /* what the flattened arrays were built from; the factor base
     * slicing changes when the siever config does, and not otherwise */
    void const * fbs = nullptr;
    gpu_engine * eng = nullptr;
    gpu_fib_task * task = nullptr;

    std::vector<gpu_fib_entry> entries;
    std::vector<gpu_fib_slice> slices;
    std::vector<slice_index_t> slice_indices;

    /* scratch, kept across special-q's so that a pass allocates
     * nothing */
    std::vector<uint32_t> counts;
    std::vector<uint32_t> host_entries;
    std::vector<uint64_t> bucket_begin;
    std::vector<uint64_t> bucket_room;

    ~las_gpu_fib_cache()
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

void las_gpu_fib_cache_deleter::operator()(las_gpu_fib_cache * p) const
{
    delete p;
}

/* }}} */
/* {{{ building the flattened factor base */

static void build_cache(las_gpu_fib_cache & C, nfs_work const & ws, int side)
{
    nfs_work::side_data const & wss(ws.sides[side]);
    fb_factorbase::slicing::part const & P = wss.fbs->get_part(1);

    C.entries.clear();
    C.slices.clear();
    C.slice_indices.clear();

    P.foreach_slice([&](auto const & sl) {
        using entry_t = typename std::decay_t<decltype(sl)>::entry_t;
        if constexpr (las_gpu_fib_takes_slice<entry_t>()) {
            gpu_fib_slice gs {};
            gs.first = uint32_t(C.entries.size());
            gs.slice_index = sl.get_index();
            slice_offset_t i_entry = 0;
            for (auto const & e: sl) {
                /* one lattice per root, which is the unit of work the
                 * kernel parallelises over */
                for (int k = 0; k < entry_t::fixed_nr_roots; k++)
                    C.entries.push_back(gpu_fib_entry {
                            e.p, e.invq, e.roots[size_t(k)], i_entry, 0});
                i_entry++;
            }
            gs.count = uint32_t(C.entries.size()) - gs.first;
            C.slices.push_back(gs);
            C.slice_indices.push_back(sl.get_index());
        }
    });

    C.eng = las_gpu::engine();
    C.task = C.eng->fib_task_create(C.entries.data(),
                                    uint32_t(C.entries.size()),
                                    C.slices.data(),
                                    uint32_t(C.slices.size()));
    C.fbs = wss.fbs;
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

   Updates are compared per (slice, bucket) as multisets: the device
   scatters with atomics, so it does not reproduce the order the host
   walks in, and nothing downstream depends on that order.
 */

static void crosscheck(las_gpu_fib_cache & C, gpu_fib_plan const & plan,
                       gpu_fib_result const & res,
                       gpu_fib_update const * out, size_t storage, int side)
{
    size_t const nb = plan.n_buckets;
    size_t const nslices = C.slices.size();

    static gpu_engine * ref = gpu_engine_cpu_create();
    static gpu_fib_task * reftask = nullptr;
    static void const * reffbs = nullptr;
    if (reffbs != C.fbs) {
        if (reftask)
            ref->fib_task_destroy(reftask);
        reftask = ref->fib_task_create(C.entries.data(),
                                       uint32_t(C.entries.size()),
                                       C.slices.data(),
                                       uint32_t(C.slices.size()));
        reffbs = C.fbs;
    }

    std::vector<gpu_fib_update> mine(storage);
    std::vector<uint32_t> rcounts(nslices * nb);
    std::vector<uint32_t> rhost(std::max<size_t>(res.nhost, 1) + 16);
    gpu_fib_result const r = ref->fib_run(reftask, plan, C.bucket_begin.data(),
                                          C.bucket_room.data(), mine.data(),
                                          rcounts.data(), rhost.data(),
                                          rhost.size());

    auto fail = [&](char const * what) {
        fprintf(stderr, "las-gpu crosscheck: side %d: %s\n", side, what);
        abort();
    };

    if (r.overflow != res.overflow)
        fail("overflow disagreement");
    if (r.nhost != res.nhost)
        fail("different number of entries handed back");
    {
        std::vector<uint32_t> a(C.host_entries.begin(),
                C.host_entries.begin() + ptrdiff_t(res.nhost));
        std::vector<uint32_t> b(rhost.begin(),
                rhost.begin() + ptrdiff_t(r.nhost));
        std::ranges::sort(a);
        std::ranges::sort(b);
        if (a != b)
            fail("different entries handed back");
    }
    if (r.overflow)
        return;

    for (size_t s = 0; s < nslices; s++)
        for (size_t b = 0; b < nb; b++)
            if (C.counts[s * nb + b] != rcounts[s * nb + b]) {
                fprintf(stderr,
                        "las-gpu crosscheck: side %d slice %zu bucket %zu:"
                        " device %u, reference %u\n",
                        side, s, b, C.counts[s * nb + b], rcounts[s * nb + b]);
                abort();
            }

    auto key = [](gpu_fib_update const & u) {
        return (uint32_t(u.x) << 16) | u.hint;
    };
    for (size_t b = 0; b < nb; b++) {
        uint64_t off = C.bucket_begin[b];
        for (size_t s = 0; s < nslices; s++) {
            uint32_t const n = C.counts[s * nb + b];
            std::vector<uint32_t> va(n), vb(n);
            for (uint32_t k = 0; k < n; k++) {
                va[k] = key(out[off + k]);
                vb[k] = key(mine[off + k]);
            }
            std::ranges::sort(va);
            std::ranges::sort(vb);
            if (va != vb) {
                fprintf(stderr,
                        "las-gpu crosscheck: side %d slice %zu bucket %zu:"
                        " %u updates differ\n", side, s, b, n);
                abort();
            }
            off += n;
        }
    }
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

void las_gpu_fib_run(worker_thread * worker, int side, nfs_work & ws,
                     nfs_aux & aux, qlattice_basis const & Q,
                     std::shared_ptr<where_am_I> /* unused, see below */)
{
    /* The where_am_I we work with is the worker's own, exactly as in
     * fill_in_buckets_toplevel_wrapper; the copy in the task's
     * arguments is only there because the ordinary tasks take one. */
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

    if (!wss.gpu_fib || wss.gpu_fib->fbs != wss.fbs ||
            wss.gpu_fib->eng != las_gpu::engine()) {
        wss.gpu_fib.reset(new las_gpu_fib_cache);
        build_cache(*wss.gpu_fib, ws, side);
    }
    las_gpu_fib_cache & C(*wss.gpu_fib);

    if (C.slices.empty())
        return;

    auto acquired = wss.reserve_BA<1, shorthint_t>();
    auto & BA = acquired.access();

    size_t const nb = BA.n_bucket;
    size_t const nslices = C.slices.size();

    /* The kernels index the per-bucket arrays by x >> log_bucket_region
     * for every x below J << logI, so the bucket array has to be at
     * least that wide. It always is -- las rounds the bucket count up,
     * and truncates J afterwards rather than before -- but an overrun
     * here would be a silent out-of-bounds atomic on the device, so it
     * is worth one comparison per special-q to be sure. */
    ASSERT_ALWAYS((uint64_t(ws.J) << ws.conf.logI) <=
                  (uint64_t(nb) << LOG_BUCKET_REGIONS[1]));

    gpu_fib_plan plan {};
    plan.logI = ws.conf.logI;
    plan.log_bucket_region = LOG_BUCKET_REGIONS[1];
    plan.J = ws.J;
    plan.n_buckets = uint32_t(nb);
    plan.basis = gpu_qlattice {Q.a0, Q.b0, Q.a1, Q.b1};

    /* Mirror of Q.is_coprime_to(): only the factors that a factor base
     * prime could possibly equal are worth listing. */
    plan.n_q_factors = 0;
    for (uint64_t f: Q.doing.prime_factors) {
        if (f <= UINT32_MAX)
            plan.q_factors[plan.n_q_factors++] = uint32_t(f);
    }

    C.counts.resize(nslices * nb);
    C.bucket_begin.resize(nb);
    C.bucket_room.resize(nb);
    for (size_t b = 0; b < nb; b++) {
        C.bucket_begin[b] = BA.gpu_bucket_begin(b);
        C.bucket_room[b] = BA.gpu_bucket_room(b);
    }
    if (C.host_entries.empty())
        C.host_entries.resize(256);

    auto * const out = reinterpret_cast<gpu_fib_update *>(BA.gpu_storage());

    gpu_fib_result res {};
    for (;;) {
        res = C.eng->fib_run(C.task, plan, C.bucket_begin.data(),
                             C.bucket_room.data(), out, C.counts.data(),
                             C.host_entries.data(), C.host_entries.size());
        if (res.nhost > C.host_entries.size()) {
            /* The kernel never drops one silently, it just tells us the
             * buffer was too small. Only ever seen on the first q. */
            C.host_entries.resize(size_t(res.nhost));
            continue;
        }
        break;
    }

    if (res.overflow) {
        throw buckets_are_full(
                bkmult_specifier::getkey<fib_update_t>(), side,
                int(res.overflow_bucket), int(res.overflow_needed),
                int(res.overflow_room));
    }

    static bool const xcheck = getenv("LAS_GPU_FIB_CROSSCHECK") != nullptr;
    if (xcheck)
        crosscheck(C, plan, res, out,
                   size_t(BA.gpu_bucket_begin(nb - 1) +
                          BA.gpu_bucket_room(nb - 1)),
                   side);

    BA.gpu_commit(C.slice_indices.data(), nslices, C.counts.data());

    for (uint64_t k = 0; k < res.nhost; k++) {
        uint32_t const idx = C.host_entries[size_t(k)];
        size_t const s = C.slice_of(idx);
        replay_one(BA, C.entries[idx], C.slices[s].slice_index, Q,
                   ws.conf.logI, ws.J, w);
    }
}

/* }}} */
