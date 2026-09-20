#ifndef CADO_LAS_NORMS_GPU_HPP
#define CADO_LAS_NORMS_GPU_HPP

#include <cstdint>
#include <vector>

#include "cado_poly.hpp"
#include "las-norms.hpp"
#include "las-siever-config.hpp"
#include "gpu/las-gpu-engine.hpp"
#include "gpu/las-gpu-plan.hpp"

/* Norm initialisation done one cell at a time, so that it can be handed
 * to a device.
 *
 * lognorm_smart walks each line and emits runs of equal values, which is
 * the right thing to do on a CPU and the wrong thing to do on a GPU: the
 * run lengths are data-dependent, so there is nothing to spread over the
 * threads. This class evaluates the very same piecewise-linear
 * approximation G, but as a pure function of (i,j), which parallelises
 * perfectly.
 *
 * The two therefore agree only up to the rounding of the run boundaries
 * -- in practice within one unit of the 8-bit log, i.e. the same kind of
 * disagreement that lognorm_smart already has with lognorm_reference.
 * tests/sieve/test_init_norms measures it.
 *
 * We inherit from lognorm_smart to reuse its constructor, which is where
 * G is computed, and to keep a working fill() when no backend has been
 * selected.
 */
struct lognorm_gpu : public lognorm_smart {
    lognorm_gpu(siever_config const & sc, cxx_cado_poly const & cpoly,
                int side, special_q_data_class auto const & Q, int logI,
                uint32_t J);

    lognorm_gpu(lognorm_gpu const &) = delete;
    lognorm_gpu & operator=(lognorm_gpu const &) = delete;
    lognorm_gpu(lognorm_gpu &&) = delete;
    lognorm_gpu & operator=(lognorm_gpu &&) = delete;
    ~lognorm_gpu() override;

    void fill(unsigned char * S, unsigned int N) const override;

    /* Fill nregions consecutive bucket regions in one go. This is the
     * call that actually makes the device worth using; fill() above is
     * the degenerate nregions == 1 case, and exists so that lognorm_gpu
     * is a drop-in lognorm_base.
     */
    void fill_many(unsigned char * S, unsigned int N0,
                   unsigned int nregions) const;

    /* false when no backend is active, in which case we silently behave
     * like lognorm_smart. */
    bool offloaded() const { return task != nullptr; }

private:
    void build_plan();

    gpu_engine * eng = nullptr;
    gpu_norm_task * task = nullptr;
    std::vector<gpu_norm_segment> segments;
    gpu_norm_plan plan {};
};

#endif /* CADO_LAS_NORMS_GPU_HPP */
