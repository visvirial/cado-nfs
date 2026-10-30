#ifndef CADO_LAS_GPU_ECM_HPP
#define CADO_LAS_GPU_ECM_HPP

#include <cstddef>
#include <cstdint>

#include <memory>

#include "ecm/facul_ecm.h"

/* ECM on the device, for the curves of the default cofactorization
 * strategy that are by far the most numerous: the MONTY12
 * parameterization (see facul_strategy_oneside::default_strategy). las
 * uses this through las-gpu-cofac.hpp.
 *
 * The point is to get, for each modulus, exactly what ecm<layer>() in
 * sieve/ecm/ecm.cpp gets: the same factor, and the same backtracking
 * flag, so that the CPU can carry on from there as if it had done the
 * work itself. The device performs the same sequence of curve operations
 * -- same parameterization, same stage 1 bytecode, same stage 2 plan --
 * so its residues are the same modulo n, and so is every decision taken
 * on them. Only the representation differs, which nothing observes.
 *
 * Moduli are odd and below 2^126, like arithxx_modredc126's.
 */

struct gpu_ecm_result {
    uint64_t f[2];      /* the factor found, 1 if none */
    int32_t bt;         /* whether backtracking was used */
    int32_t pad;
};

class gpu_ecm_runner {
public:
    virtual ~gpu_ecm_runner() = default;

    /* Run the plan, whose parameterization must be MONTY12, on each of
     * the count moduli n[2*i], n[2*i+1] (low, high), and write the outcome
     * to out[i]. Blocks. The plan must stay alive and unchanged as long as
     * the runner does: it is put on the device once.
     *
     * Thread-safe; the callers take turns. Returns false, with nothing
     * written, when the device has no memory for this: other processes
     * may share it, and the caller has the CPU to fall back on. */
    virtual bool run(ecm_plan_t const & plan, uint64_t const * n, size_t count,
                     gpu_ecm_result * out) = 0;
};

/* nullptr if there is no CUDA backend, or no such device. */
std::unique_ptr<gpu_ecm_runner> gpu_ecm_runner_create(int device);

#endif /* CADO_LAS_GPU_ECM_HPP */
