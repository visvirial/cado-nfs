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

/* Human-readable name of the active backend, "none" if there is none. */
std::string describe();

} // namespace las_gpu

#endif /* CADO_LAS_GPU_SELECT_HPP */
