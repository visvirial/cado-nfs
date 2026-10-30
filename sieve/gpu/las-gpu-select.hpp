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
 *   "auto"      "cuda" if that works, "none" otherwise (las's default)
 *   "none"      offloading disabled
 *   "cpu"       the portable host backend -- same arithmetic, no device.
 *               A reference for testing, much slower than "none".
 *   "cuda"      first usable CUDA device
 *   "cuda:<n>"  CUDA device number n
 * Throws std::runtime_error on an unknown spec, or when "cuda" or
 * "cuda:<n>" names a device that cannot be used. A device that is there
 * but has no memory free -- other processes hold it -- is not an error:
 * offloading starts out suspended, and the device is opened once a
 * suspension runs out and it has memory free (see suspend()).
 */
void select(std::string const & spec);

/* nullptr when no backend was selected -- or not yet: a CUDA device
 * that has no memory free for las is opened later, once it has. */
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

/* Whether stage s is to be offloaded right now. */
bool stage_enabled(stage s);

/* Whether it is to be offloaded at all, at least once offloading is no
 * longer suspended -- what to go by when deciding on things that last
 * the whole run. */
bool stage_selected(stage s);

/* Stop offloading for a while, e.g. because the device ran out of
 * memory -- most likely held by another process, which may give it
 * back. stage_enabled() returns false until the time is up; the time
 * doubles with every failure in a row, from 30 seconds up to half an
 * hour. The engine itself stays, so that passes already in flight can
 * complete. Prints `why`, once per suspension. Thread-safe. */
void suspend(char const * why);

/* A device pass went through: the next suspension, if any, is a short
 * one again. Thread-safe, and cheap. */
void pass_succeeded();

/* Human-readable name of the active backend, "none" if there is none. */
std::string describe();

} // namespace las_gpu

#endif /* CADO_LAS_GPU_SELECT_HPP */
