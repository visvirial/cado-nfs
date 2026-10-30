#include "cado.h" // IWYU pragma: keep

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <atomic>
#include <limits>
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
#include "las-info.hpp"
#include "las-norms.hpp"
#include "las-plattice.hpp"
#include "las-qlattice.hpp"
#include "las-siever-config.hpp"
#include "las-threads-work-data.hpp"
#include "las-auxiliary-data.hpp"
#include "las-where-am-i.hpp"
#include "macros.h"
#include "smallsieve.hpp"
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

/* Where every lattice's walk stopped, for chunked passes. Only one
 * special-q per side is ever chunked at a time -- chunked passes are
 * never prefetched -- so the images of a side share one, rather than
 * each keeping a gigabyte of their own at c270 sizes. It fits any task
 * with the same number of entries, which the slicings of a side have. */
struct las_gpu_fib_walk {
    gpu_engine * eng = nullptr;
    gpu_fib_state * state = nullptr;
    size_t nentries = 0;

    las_gpu_fib_walk(gpu_engine * eng, size_t nentries)
        : eng(eng), nentries(nentries) {}
    las_gpu_fib_walk(las_gpu_fib_walk const &) = delete;
    las_gpu_fib_walk & operator=(las_gpu_fib_walk const &) = delete;
    ~las_gpu_fib_walk()
    {
        if (state)
            eng->fib_state_destroy(state);
    }
};

/* The flattened form of one factor base slicing, and its device copy. */
struct las_gpu_fib_image {
    gpu_engine * eng = nullptr;
    gpu_fib_task * task = nullptr;

    std::vector<gpu_fib_entry> entries;
    std::vector<gpu_fib_slice> slices;
    std::vector<slice_index_t> slice_indices;
    std::vector<double> weights;

    /* when get_image() last handed it out, for eviction */
    uint64_t last_used = 0;

    las_gpu_fib_image() = default;
    las_gpu_fib_image(las_gpu_fib_image const &) = delete;
    las_gpu_fib_image & operator=(las_gpu_fib_image const &) = delete;

    /* Walk state for chunked passes, created on first use; see
     * las_gpu_fib_walk. Set by get_image(). */
    std::shared_ptr<las_gpu_fib_walk> walk;

    gpu_fib_state * get_state()
    {
        if (!walk->state && task)
            walk->state = eng->fib_state_create(task);
        return walk->state;
    }

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
     * factor base on most special-q's.
     *
     * At c270 sizes an image is a gigabyte or so on the device, even
     * with the engine sharing the entries between slicings, and the
     * device may be shared with other processes. So they are kept only
     * as long as there is room: upload() drops the least recently used
     * ones. The passes that use an image hold it, so that one dropped
     * from here lives on until they are done. */
    gpu_engine * eng = nullptr;
    std::map<void const *, std::shared_ptr<las_gpu_fib_image>> images;

    std::shared_ptr<las_gpu_fib_walk> walk;

    /* The image that prepare_for_new_q() chose for the special-q being
     * sieved, held from then on so that nothing evicts it before
     * las_gpu_fib_start() launches its passes. */
    std::shared_ptr<las_gpu_fib_image> current;
};

void las_gpu_fib_cache_deleter::operator()(las_gpu_fib_cache * p) const
{
    delete p;
}

/* }}} */
/* {{{ building the flattened factor base */

static std::shared_ptr<las_gpu_fib_image>
build_image(fb_factorbase::slicing const & fbs)
{
    auto I = std::make_shared<las_gpu_fib_image>();
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

    gpu_fib_pass * const rp = ref->fib_launch(reftask, plan, nullptr);
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

bool las_gpu_fib_usable_for(siever_config const & conf MAYBE_UNUSED,
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
    if (Q.sublat.m > 1 || conf.sublat_bound > 1)
        return false;
    /* The kernel filters the primes dividing the special-q against a
     * small fixed-size list, like Q.is_coprime_to() does. A q with more
     * factors than that is not worth a special case. */
    if (Q.doing.prime_factors.size() > 8)
        return false;
    return true;
#endif
}

bool las_gpu_fib_usable(nfs_work const & ws, qlattice_basis const & Q)
{
    return ws.toplevel == 1 && las_gpu_fib_usable_for(ws.conf, Q);
}

/* {{{ chunks
 *
 * A pass handles the updates of a range of buckets at once, indexes
 * them with 32-bit offsets, and keeps one counter per (bucket, slice).
 * For a small sieve area, one pass covers everything. A large one does
 * not fit: at I=17, as for a c270, one side produces about 5e9 updates,
 * 20GB, and the grid would have 2e8 counters. So the area is cut into
 * chunks of buckets, each walked from where the previous one stopped
 * (see gpu_fib_walk_until), and handed to the bucket arrays one after
 * the other as the bucket regions are processed.
 *
 * Chunks aim at FIB_CHUNK_TARGET updates, 256MB: enough to keep the
 * device busy, little enough that the three in flight per side -- one
 * being processed, one waited for, one being computed -- fit beside the
 * factor base on the device. They are a power of two of at least
 * fib_chunk_min() buckets, a multiple of the batches in which
 * process_many_bucket_regions() works at toplevel 1 -- 1 <<
 * LOG_BUCKET_REGION_step regions -- so that a batch never straddles two
 * chunks.
 */
static constexpr double FIB_CHUNK_TARGET = double(UINT64_C(1) << 26);
static uint32_t fib_chunk_min()
{
    return std::max(UINT32_C(1024), UINT32_C(1) << LOG_BUCKET_REGION_step);
}

/* Buckets per chunk, or 0 when the device path cannot take this area at
 * all. nb is a power of two, as las makes it.
 *
 * The updates are spread evenly over the cells of the area, but the area
 * is only the first J rows: with a small J, the buckets past it are
 * empty, and the ones before it as full as ever. So a chunk is sized by
 * the buckets of it that are in use, not by its share of nb -- which
 * would underestimate it by nb over the buckets in use, nearly three
 * times for a c270 special-q with J a third of the maximum. */
static uint32_t chunk_buckets(double weight, int logI, uint32_t J,
                              uint32_t nb, size_t nslices)
{
    int const logB = LOG_BUCKET_REGIONS[1];
    double const per_bucket = weight * double(UINT64_C(1) << logB);
    uint64_t const used = std::min<uint64_t>(nb,
            iceildiv(uint64_t(J) << logI, UINT64_C(1) << logB));
    auto expected = [&](uint32_t K) {
        return per_bucket * double(std::min<uint64_t>(K, used));
    };
    uint32_t K = nb;
    while (K > fib_chunk_min() && !(K & 1) &&
            (expected(K) > FIB_CHUNK_TARGET ||
             uint64_t(K) * nslices >= (UINT64_C(1) << 24)))
        K /= 2;
    /* The limits leave a factor of two of margin on the expected count,
     * which the actual one fluctuates around by far less. */
    if (expected(K) >= double(UINT64_C(1) << 31) ||
            uint64_t(K) * nslices >= (UINT64_C(1) << 24))
        return 0;
    return K;
}

static double total_weight(las_gpu_fib_image const & I)
{
    return std::accumulate(I.weights.begin(), I.weights.end(), 0.0);
}

/* }}} */

/* {{{ the images on the device */

/* What is left on the device beside the images, for the passes'
 * scratch, which the engine grows on demand. */
static constexpr size_t FIB_DEVICE_HEADROOM = size_t(1) << 30;

/* The image of this slicing, built on the host if need be -- not
 * necessarily on the device yet, see upload(). */
static std::shared_ptr<las_gpu_fib_image> const &
get_image(nfs_work::side_data & wss, fb_factorbase::slicing const * fbs)
{
    static std::atomic<uint64_t> stamp {0};
    if (!wss.gpu_fib || wss.gpu_fib->eng != las_gpu::engine()) {
        wss.gpu_fib.reset(new las_gpu_fib_cache);
        wss.gpu_fib->eng = las_gpu::engine();
    }
    auto & slot = wss.gpu_fib->images[fbs];
    if (!slot) {
        slot = build_image(*fbs);
        auto & walk(wss.gpu_fib->walk);
        if (!walk || walk->nentries != slot->entries.size())
            walk = std::make_shared<las_gpu_fib_walk>(
                    slot->eng, slot->entries.size());
        slot->walk = walk;
    }
    slot->last_used = ++stamp;
    return slot;
}

/* Drop the least recently used image on the device, on either side,
 * other than `keep`. Only images that nobody else holds go: not the
 * ones that passes in flight or prefetched use, nor the current
 * special-q's. False when there is none. */
static bool evict_one(nfs_work & ws, las_gpu_fib_image const & keep)
{
    decltype(las_gpu_fib_cache::images) * where = nullptr;
    decltype(las_gpu_fib_cache::images)::iterator victim;
    for (auto & wss: ws.sides) {
        if (!wss.gpu_fib)
            continue;
        auto & images(wss.gpu_fib->images);
        for (auto it = images.begin(); it != images.end(); ++it) {
            auto const & J(it->second);
            if (J.get() == &keep || !J->task || J.use_count() > 1)
                continue;
            if (!where || J->last_used < victim->second->last_used) {
                where = &images;
                victim = it;
            }
        }
    }
    if (!where)
        return false;
    where->erase(victim);
    return true;
}

/* Put the image on the device if it is not there yet, dropping others
 * to make room as long as it does not fit, and then to leave room for
 * the scratch. False when it does not fit even so.
 *
 * How much room an image takes is up to the engine, which shares what
 * it can between the slicings of a factor base -- so we just try. */
static bool upload(nfs_work & ws, las_gpu_fib_image & I)
{
    if (I.task || I.slices.empty())
        return true;
    for (;;) {
        I.task = I.eng->fib_task_create(I.entries.data(),
                                        uint32_t(I.entries.size()),
                                        I.slices.data(),
                                        uint32_t(I.slices.size()));
        if (I.task)
            break;
        if (!evict_one(ws, I))
            return false;
    }
    while (I.eng->available_memory() < FIB_DEVICE_HEADROOM &&
            evict_one(ws, I))
        ;
    return true;
}

/* }}} */

double las_gpu_fib_weight(nfs_work & ws, int side,
                          qlattice_basis const & Q)
{
    nfs_work::side_data & wss(ws.sides[side]);
    if (wss.gpu_fib)
        wss.gpu_fib->current.reset();
    if (wss.no_fb() || !las_gpu_fib_usable(ws, Q))
        return 0;
    std::shared_ptr<las_gpu_fib_image> const I = get_image(wss, wss.fbs);
    double const w = total_weight(*I);
    if (I->slices.empty() ||
            !chunk_buckets(w, ws.conf.logI, ws.J,
                           uint32_t(ws.nb_buckets[1]), I->slices.size()))
        return 0;
    /* Better found out now than once the bucket arrays are sized for
     * the device: then the special-q would have to be done again. */
    if (!upload(ws, *I)) {
        las_gpu::suspend("the factor base does not fit on the device");
        return 0;
    }
    wss.gpu_fib->current = I;
    return w;
}

/* The device passes of one special-q on one side: one per chunk. It
 * outlives the bucket filling -- the bucket arrays point into the
 * passes' output -- and is held by the side data until the next
 * special-q.
 *
 * Chunk c+1 is launched as soon as chunk c has been waited for, which
 * is what the walk state requires; it is waited for in the background
 * while chunk c's bucket regions are processed; and chunk c is released
 * once they are done. */
struct las_gpu_fib_inflight {
    std::shared_ptr<las_gpu_fib_image> held;
    las_gpu_fib_image & I;
    gpu_fib_plan plan;          /* chunk_begin and chunk_count unset */
    void const * fbs;
    int side;
    uint32_t K;                 /* buckets per chunk */
    uint32_t nchunks;
    gpu_fib_state * state = nullptr;

    struct chunk {
        gpu_fib_pass * pass = nullptr;
        gpu_fib_output out {};
        std::once_flag waited;
        bool queued = false;    /* background wait queued */
    };
    std::vector<std::unique_ptr<chunk>> chunks;

    /* the chunk whose updates the bucket arrays currently see, and the
     * array they are registered with */
    int current = -1;
    bucket_array_t<1, shorthint_t> * BA = nullptr;

    las_gpu_fib_inflight(std::shared_ptr<las_gpu_fib_image> held,
                         gpu_fib_plan const & plan,
                         void const * fbs, int side, uint32_t K)
        : held(std::move(held)), I(*this->held), plan(plan), fbs(fbs),
          side(side), K(K)
    {
        int const logB = plan.log_bucket_region;
        /* the buckets past J << logI are empty, and never processed */
        uint64_t const used = std::min<uint64_t>(plan.n_buckets,
                iceildiv(uint64_t(plan.J) << plan.logI,
                         UINT64_C(1) << logB));
        nchunks = uint32_t(std::max<uint64_t>(1, iceildiv(used, K)));
        for (uint32_t c = 0; c < nchunks; c++)
            chunks.emplace_back(std::make_unique<chunk>());
        if (nchunks > 1 && !(state = I.get_state()))
            return;     /* no memory for it: chunk 0 has no pass */
        launch(0);
    }
    las_gpu_fib_inflight(las_gpu_fib_inflight const &) = delete;
    las_gpu_fib_inflight & operator=(las_gpu_fib_inflight const &) = delete;

    ~las_gpu_fib_inflight()
    {
        for (uint32_t c = 0; c < nchunks; c++)
            release(c);
    }

    gpu_fib_plan plan_for(uint32_t c) const
    {
        gpu_fib_plan p = plan;
        p.chunk_begin = c * K;
        p.chunk_count = std::min(K, plan.n_buckets - c * K);
        return p;
    }

    void launch(uint32_t c)
    {
        if (I.task)
            chunks[c]->pass = I.eng->fib_launch(I.task, plan_for(c), state);
    }

    /* Thread-safe; must be called in increasing order of c, which the
     * single caller of each -- collect(), then las_gpu_fib_advance()
     * and the background tasks it queues -- ensures. */
    gpu_fib_output const & wait(uint32_t c)
    {
        chunk & C(*chunks[c]);
        std::call_once(C.waited, [&]() {
            if (C.pass)
                C.out = I.eng->fib_wait(C.pass);
            if (!C.out.ok)
                return;
            las_gpu::pass_succeeded();
            static bool const xcheck =
                getenv("LAS_GPU_FIB_CROSSCHECK") != nullptr;
            if (xcheck)
                crosscheck(I, plan_for(c), C.out, side);
            if (c + 1 < nchunks)
                launch(c + 1);
        });
        return C.out;
    }

    void release(uint32_t c)
    {
        if (chunks[c]->pass)
            I.eng->fib_release(chunks[c]->pass);
        chunks[c]->pass = nullptr;
    }

    void show(uint32_t c)
    {
        gpu_fib_output const & out(chunks[c]->out);
        BA->add_external_slices(I.slice_indices.data(), 0, out.nslices,
                reinterpret_cast<fib_update_t const *>(out.updates),
                out.offsets, out.nslices,
                size_t(c) * K, size_t(c) * K + out.n_buckets);
        current = int(c);
    }
};

/* The device ran out of memory, most likely because other processes
 * share it -- or, in principle, a pass was too large for its 32-bit
 * offsets, which the backend checks. The bucket arrays have no room for
 * what the device was to do, so this special-q has to start over: with
 * offloading suspended, prepare_for_new_q() sizes them for the whole
 * part.
 * buckets_are_full is what makes las do that; at ratio 1 it grows the
 * multiplier by its 5% margin only. Relations already found are not
 * printed twice: las remembers them across attempts at a special-q. */
static buckets_are_full device_failed(int side)
{
    las_gpu::suspend("a device pass could not be completed (see above)");
    return buckets_are_full(bkmult_specifier::getkey<fib_update_t>(), side,
                            0, 1, 1);
}

/* Wait for the first chunk, then hand it to a bucket array -- any one:
 * apply_one_bucket and the purge walk every array of the side, and the
 * device's updates take no room in them. The lattices that the device
 * handed back are replayed into the same array, over the whole area. */
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
    gpu_fib_output const & out(F->wait(0));
    if (!out.ok)
        throw device_failed(side);

    auto acquired = wss.reserve_BA<1, shorthint_t>();
    auto & BA = acquired.access();
    ASSERT_ALWAYS(BA.n_bucket == F->plan.n_buckets);

    F->BA = &BA;
    F->show(0);

    for (uint64_t k = 0; k < out.nhost; k++) {
        uint32_t const idx = out.host_entries[k];
        size_t const s = I.slice_of(idx);
        replay_one(BA, I.entries[idx], I.slices[s].slice_index, Q,
                   ws.conf.logI, ws.J, w);
    }
}

bool las_gpu_fib_advance(nfs_work & ws, thread_pool & pool, int region)
{
    for (auto & wss: ws.sides) {
        auto F = std::static_pointer_cast<las_gpu_fib_inflight>(
                wss.gpu_fib_pass);
        if (!F || F->nchunks <= 1 || !F->BA)
            continue;
        uint32_t const c = uint32_t(region) / F->K;
        if (c >= F->nchunks)
            continue;
        if (int(c) != F->current) {
            /* the regions of the previous chunk are all done: the
             * caller drains the pool between batches */
            F->BA->clear_external_slices();
            if (F->current >= 0)
                F->release(uint32_t(F->current));
            if (!F->wait(c).ok) {
                F->current = -1;
                return false;
            }
            F->show(c);
        }
        if (c + 1 < F->nchunks && !F->chunks[c + 1]->queued) {
            /* Get the next chunk back from the device while this one's
             * regions are processed; ahead of them in the queue, so
             * that it is not left for last. */
            F->chunks[c + 1]->queued = true;
            pool.add_task(thread_pool::QUEUE_GENERIC,
                    std::numeric_limits<double>::max(),
                    [F, c](worker_thread *) { F->wait(c + 1); });
        }
    }
    return true;
}

void las_gpu_fib_check_failure(nfs_work const & ws)
{
    for (int side = 0; side < int(ws.sides.size()); side++) {
        auto F = std::static_pointer_cast<las_gpu_fib_inflight>(
                ws.sides[side].gpu_fib_pass);
        if (F && F->BA && F->current < 0)
            throw device_failed(side);
    }
}

static gpu_fib_plan make_plan(int logI, uint32_t J, uint32_t nb,
                              qlattice_basis const & Q)
{
    /* The kernels index the per-bucket arrays by x >> log_bucket_region
     * for every x below J << logI, so the bucket arrays have to be at
     * least that wide. They always are -- las rounds the bucket count
     * up, and truncates J afterwards rather than before -- but an
     * overrun here would be a silent out-of-bounds atomic on the
     * device, so it is worth one comparison per special-q to be sure. */
    ASSERT_ALWAYS((uint64_t(J) << logI) <=
                  (uint64_t(nb) << LOG_BUCKET_REGIONS[1]));

    gpu_fib_plan plan {};
    plan.logI = logI;
    plan.log_bucket_region = LOG_BUCKET_REGIONS[1];
    plan.J = J;
    plan.n_buckets = nb;
    plan.basis = gpu_qlattice {Q.a0, Q.b0, Q.a1, Q.b1};

    /* Mirror of Q.is_coprime_to(): only the factors that a factor base
     * prime could possibly equal are worth listing. */
    plan.n_q_factors = 0;
    for (uint64_t f: Q.doing.prime_factors) {
        if (f <= UINT32_MAX)
            plan.q_factors[plan.n_q_factors++] = uint32_t(f);
    }
    return plan;
}

static bool same_plan(gpu_fib_plan const & a, gpu_fib_plan const & b)
{
    if (a.logI != b.logI || a.log_bucket_region != b.log_bucket_region ||
            a.J != b.J || a.n_buckets != b.n_buckets ||
            a.basis.a0 != b.basis.a0 || a.basis.b0 != b.basis.b0 ||
            a.basis.a1 != b.basis.a1 || a.basis.b1 != b.basis.b1 ||
            a.n_q_factors != b.n_q_factors)
        return false;
    for (int i = 0; i < a.n_q_factors; i++)
        if (a.q_factors[i] != b.q_factors[i])
            return false;
    return true;
}

double las_gpu_fib_start(nfs_work & ws, nfs_aux & aux,
                         qlattice_basis const & Q, int side,
                         thread_pool & pool)
{
    nfs_work::side_data & wss(ws.sides[side]);

    /* las_gpu_fib_weight() put it on the device, and has held it since */
    std::shared_ptr<las_gpu_fib_image> const held = get_image(wss, wss.fbs);
    las_gpu_fib_image & I(*held);

    /* Even when offloading has been suspended since prepare_for_new_q()
     * decided on it, we go ahead: the bucket arrays were sized for it.
     * At worst the launch fails, and collect() has the special-q
     * redone. */
    if (I.slices.empty())
        return 0;

    uint32_t const nb = uint32_t(ws.nb_buckets[1]);
    uint32_t const K = chunk_buckets(total_weight(I), ws.conf.logI, ws.J,
                                     nb, I.slices.size());
    /* las_gpu_fib_weight() said yes, with the same numbers */
    ASSERT_ALWAYS(K);

    gpu_fib_plan const plan = make_plan(ws.conf.logI, ws.J, nb, Q);

    /* Most of the time las_gpu_fib_prefetch() has already launched this
     * very pass, while the previous special-q was being sieved. It was
     * computed from a sieve area and a slicing chosen ahead of time, so
     * we only take it if they turn out to be the ones we have now; the
     * plan and the slicing are all the device ever saw. A prefetched
     * pass that does not match is left alone rather than dropped: when
     * a special-q is retried after buckets_are_full, the one in store
     * is for the next special-q, not for this one. */
    std::shared_ptr<las_gpu_fib_inflight> F;
    if (auto P = std::static_pointer_cast<las_gpu_fib_inflight>(
                wss.gpu_fib_prefetched)) {
        if (P->fbs == wss.fbs && &P->I == &I && P->K == K &&
                same_plan(P->plan, plan)) {
            F = std::move(P);
            wss.gpu_fib_prefetched.reset();
        }
    }
    if (!F)
        F = std::make_shared<las_gpu_fib_inflight>(held, plan, wss.fbs,
                                                   side, K);
    wss.gpu_fib_pass = F;

    /* Cost 0, so that the pool gets to it last: by then the device has
     * had all the time the ordinary tasks took. */
    pool.add_task(thread_pool::QUEUE_GENERIC, 0, collect,
                  side, std::ref(ws), std::ref(aux), std::cref(Q), F);

    return total_weight(I);
}

void las_gpu_fib_prefetch(las_info & las, nfs_work & ws,
                          siever_config const & conf,
                          qlattice_basis const & Q, uint32_t J)
{
    if (!las_gpu_fib_usable_for(conf, Q))
        return;

    int const nsides = las.cpoly.nsides();

    /* What prepare_for_new_q() and compute_toplevel_and_buckets() will
     * decide once this special-q is actually sieved. Any divergence is
     * caught by las_gpu_fib_start(), which then ignores the pass. */
    std::vector<fb_factorbase::slicing const *> fbs(nsides, nullptr);
    int toplevel = -1;
    for (int side = 0; side < nsides; side++) {
        if (las.no_fb(side))
            continue;
        lognorm_base const L(conf, las.cpoly, side, Q, conf.logI, J);
        fb_factorbase::key_type fbK = conf.instantiate_thresholds(side);
        fbK.scale = L.scale;
        fbK.nb_threads = las.number_of_threads_per_subjob();
        fbs[side] = las.get_factorbase_slicing(side, fbK);
        toplevel = std::max(toplevel, fbs[side]->get_toplevel());
    }
    if (toplevel != 1)
        return;
    uint32_t const nb = uint32_t(iceildiv(UINT64_C(1) << conf.logA,
                                          BUCKET_REGIONS[1]));
    if (nb <= 1)
        return;

    for (int side = 0; side < nsides; side++) {
        nfs_work::side_data & wss(ws.sides[side]);
        wss.gpu_fib_prefetched.reset();
        if (!fbs[side])
            continue;
        std::shared_ptr<las_gpu_fib_image> const held =
            get_image(wss, fbs[side]);
        las_gpu_fib_image & I(*held);
        if (I.slices.empty())
            continue;
        /* Only a pass over the whole area at once: the chunks of a
         * large one share a walk state, which the special-q being
         * sieved is still using. Those are few and long anyway, so
         * there is little to win. */
        if (chunk_buckets(total_weight(I), conf.logI, J, nb,
                          I.slices.size()) != nb)
            continue;
        /* No room is no harm here: nothing is sized for this pass yet,
         * and prepare_for_new_q() tries again, with the device as it
         * is by then, and decides. */
        if (!upload(ws, I))
            return;
        auto P = std::make_shared<las_gpu_fib_inflight>(
                held, make_plan(conf.logI, J, nb, Q), fbs[side], side, nb);
        if (!P->chunks[0]->pass)
            return;
        wss.gpu_fib_prefetched = std::move(P);
    }
}

/* }}} */
