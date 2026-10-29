#include "cado.h" // IWYU pragma: keep

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <vector>

#include "las-gpu-engine.hpp"
#include "las-gpu-fib-kernels.hpp"
#include "las-gpu-fib-plan.hpp"
#include "las-gpu-kernels.hpp"
#include "las-gpu-plan.hpp"

/* Host backend. It runs the very same per-cell functions as the CUDA
 * kernels, one cell at a time, and is therefore the reference against
 * which the device code is checked -- including on machines that have
 * no GPU at all.
 *
 * It is not meant to be fast. lognorm_smart, which emits runs instead of
 * individual cells, beats it comfortably on a single core, and that is
 * expected: the per-cell formulation only pays off once there are
 * thousands of threads to hide the arithmetic behind.
 */

gpu_engine::~gpu_engine() = default;

namespace {

struct cpu_norm_task {
    gpu_norm_plan plan;
    std::vector<gpu_norm_segment> segments;
};

struct cpu_survivor_task {
    std::vector<gpu_jdiv_entry> jdiv;
};

struct cpu_fib_task {
    std::vector<gpu_fib_entry> entries;
    std::vector<gpu_fib_slice> slices;
    /* write cursors, one per (slice, bucket); kept here so that a pass
     * does not have to allocate */
    std::vector<uint64_t> cursor;
};

struct cpu_engine : public gpu_engine {
    char const * name() const override { return "cpu"; }

    gpu_norm_task * norm_task_create(gpu_norm_plan const & plan,
                                     gpu_norm_segment const * segments,
                                     int nsegments) override
    {
        auto * t = new cpu_norm_task;
        t->plan = plan;
        t->segments.assign(segments, segments + nsegments);
        t->plan.segments = t->segments.data();
        t->plan.nsegments = nsegments;
        return reinterpret_cast<gpu_norm_task *>(t);
    }

    void norm_task_destroy(gpu_norm_task * task) override
    {
        delete reinterpret_cast<cpu_norm_task *>(task);
    }

    void norm_fill(gpu_norm_task * task, unsigned char * S, unsigned int N0,
                   unsigned int nregions) override
    {
        auto const & t = *reinterpret_cast<cpu_norm_task *>(task);
        size_t const region_size = size_t(1) << t.plan.log_bucket_region;

        for (unsigned int r = 0; r < nregions; r++) {
            unsigned char * const Sr = S + r * region_size;
            auto const geo = gpu_region_geometry_for(
                t.plan.logI, t.plan.log_bucket_region, N0 + r);
            for (size_t x = 0; x < region_size; x++)
                Sr[x] = gpu_norm_cell(t.plan, geo, uint32_t(x));
        }
    }

    gpu_survivor_task * survivor_task_create(gpu_jdiv_entry const * jdiv,
                                             uint32_t jdiv_size,
                                             size_t /* max_region_size */) override
    {
        auto * t = new cpu_survivor_task;
        t->jdiv.assign(jdiv, jdiv + jdiv_size);
        return reinterpret_cast<gpu_survivor_task *>(t);
    }

    void survivor_task_destroy(gpu_survivor_task * task) override
    {
        delete reinterpret_cast<cpu_survivor_task *>(task);
    }

    size_t survivor_search(gpu_survivor_task * task, unsigned char * S0,
                           unsigned char * S1, size_t region_size,
                           gpu_survivor_region const & region, uint32_t * out,
                           size_t out_capacity) override
    {
        auto const & t = *reinterpret_cast<cpu_survivor_task *>(task);

        gpu_survivor_tables T {};
        T.jdiv = t.jdiv.data();
        T.jdiv_size = uint32_t(t.jdiv.size());

        unsigned char * const Sf = region.side_of_Sf == 0 ? S0 : S1;
        size_t n = 0;

        for (size_t x = 0; x < region_size; x++) {
            unsigned char const s0 = S0 ? S0[x] : 255;
            unsigned char const s1 = S1 ? S1[x] : 255;
            unsigned char sf_out;
            bool const alive =
                gpu_survivor_cell(region, T, s0, s1, uint32_t(x), sf_out);
            Sf[x] = sf_out;
            if (alive) {
                if (n < out_capacity)
                    out[n] = uint32_t(x);
                n++;
            }
        }
        return n;
    }

    /* --- bucket sieving ------------------------------------------ */

    gpu_fib_task * fib_task_create(gpu_fib_entry const * entries,
                                   uint32_t nentries,
                                   gpu_fib_slice const * slices,
                                   uint32_t nslices) override
    {
        auto * t = new cpu_fib_task;
        t->entries.assign(entries, entries + nentries);
        t->slices.assign(slices, slices + nslices);
        return reinterpret_cast<gpu_fib_task *>(t);
    }

    void fib_task_destroy(gpu_fib_task * task) override
    {
        delete reinterpret_cast<cpu_fib_task *>(task);
    }

    gpu_fib_result fib_run(gpu_fib_task * task, gpu_fib_plan const & plan,
                           uint64_t const * bucket_begin,
                           uint64_t const * bucket_room,
                           gpu_fib_update * out, uint32_t * counts,
                           uint32_t * host_entries,
                           uint64_t host_capacity) override
    {
        auto & t = *reinterpret_cast<cpu_fib_task *>(task);

        uint32_t const nb = plan.n_buckets;
        size_t const nslices = t.slices.size();
        int const logB = plan.log_bucket_region;
        uint32_t const bmask = (UINT32_C(1) << logB) - 1;

        gpu_fib_fence const F = gpu_fib_fence_for(plan.logI, plan.J);

        gpu_fib_result res {};
        std::fill_n(counts, nslices * nb, uint32_t(0));

        /* Pass 1: how many updates each (slice, bucket) pair gets. The
         * device does the same thing, and for the same reason: the
         * per-slice boundaries have to be known before anything can be
         * written, and re-walking is cheaper than remembering every
         * position. */
        for (size_t s = 0; s < nslices; s++) {
            auto const & sl = t.slices[s];
            uint32_t * const row = counts + s * nb;
            for (uint32_t k = 0; k < sl.count; k++) {
                gpu_fib_entry const & e = t.entries[sl.first + k];
                if (!gpu_fib_entry_is_sieved(plan, e.p))
                    continue;
                gpu_ple ple {};
                int const kind = gpu_fib_prepare(&ple, e, plan.basis, F,
                                                 plan.logI);
                if (kind == GPU_FIB_HOST) {
                    if (res.nhost < host_capacity)
                        host_entries[res.nhost] = sl.first + k;
                    res.nhost++;
                    continue;
                }
                gpu_fib_walk(ple, kind, F, [&](uint64_t x) {
                    row[x >> logB]++;
                });
            }
        }

        /* Lay the slices out inside each bucket, in slice order, and
         * check that they fit. */
        t.cursor.assign(nslices * nb, 0);
        for (uint32_t b = 0; b < nb; b++) {
            uint64_t used = 0;
            for (size_t s = 0; s < nslices; s++) {
                t.cursor[s * nb + b] = bucket_begin[b] + used;
                used += counts[s * nb + b];
            }
            if (used > bucket_room[b] && !res.overflow) {
                res.overflow = true;
                res.overflow_bucket = b;
                res.overflow_needed = used;
                res.overflow_room = bucket_room[b];
            }
        }
        if (res.overflow || res.nhost > host_capacity)
            return res;

        /* Pass 2: write them. */
        for (size_t s = 0; s < nslices; s++) {
            auto const & sl = t.slices[s];
            uint64_t * const cur = t.cursor.data() + s * nb;
            for (uint32_t k = 0; k < sl.count; k++) {
                gpu_fib_entry const & e = t.entries[sl.first + k];
                if (!gpu_fib_entry_is_sieved(plan, e.p))
                    continue;
                gpu_ple ple {};
                int const kind = gpu_fib_prepare(&ple, e, plan.basis, F,
                                                 plan.logI);
                if (kind == GPU_FIB_HOST)
                    continue;
                uint16_t const hint = e.hint;
                gpu_fib_walk(ple, kind, F, [&](uint64_t x) {
                    out[cur[x >> logB]++] =
                        gpu_fib_update {hint, uint16_t(x & bmask)};
                });
            }
        }

        return res;
    }
};

} // namespace

gpu_engine * gpu_engine_cpu_create()
{
    return new cpu_engine;
}

#ifndef HAVE_CUDA
gpu_engine * gpu_engine_cuda_create(int)
{
    return nullptr;
}
bool gpu_engine_cuda_compiled()
{
    return false;
}
#endif
