#include "cado.h" // IWYU pragma: keep

#include <cstddef>
#include <cstdint>
#include <vector>

#include "las-gpu-engine.hpp"
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
