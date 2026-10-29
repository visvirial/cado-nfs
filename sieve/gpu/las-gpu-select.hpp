#ifndef CADO_LAS_GPU_SELECT_HPP
#define CADO_LAS_GPU_SELECT_HPP

#include <string>

#include "las-gpu-engine.hpp"

/* Process-wide choice of the backend that the offloaded sieve stages
 * use. Set it once, from the command line, before any special-q is
 * processed: the tasks created by lognorm_gpu and by the survivor
 * search capture the engine at construction time.
 */
namespace las_gpu {

/* Accepted specs:
 *   "none"      offloading disabled (the default)
 *   "cpu"       the portable host backend -- same arithmetic, no device
 *   "cuda"      first usable CUDA device
 *   "cuda:<n>"  CUDA device number n
 *   "auto"      "cuda" if that works, "cpu" otherwise
 * Throws std::runtime_error on an unknown spec or an unusable device.
 */
void select(std::string const & spec);

/* nullptr when no backend was selected. */
gpu_engine * engine();

/* Which sieve stages the backend is used for.
 *
 * Offloading is not a win everywhere. Bucket sieving is about half of
 * las's time and parallelises beautifully, so it pays off handsomely.
 * Norm initialisation and the survivor search are about one and a half
 * percent between them, and the host code for both is very good --
 * lognorm_smart emits runs rather than cells, the survivor search is
 * vectorised -- so a device loses to it on everything but the very
 * largest sieve areas. Hence the default: buckets only.
 *
 * Accepted specs: a comma-separated list of "norms", "survivors",
 * "buckets", or "all", or "none".
 * Throws std::runtime_error on an unknown stage name.
 */
enum stage : unsigned {
    STAGE_NORMS = 1u,
    STAGE_SURVIVORS = 2u,
    STAGE_BUCKETS = 4u,
};

void select_stages(std::string const & spec);
bool stage_enabled(stage s);

/* Human-readable name of the active backend, "none" if there is none. */
std::string describe();

} // namespace las_gpu

#endif /* CADO_LAS_GPU_SELECT_HPP */
