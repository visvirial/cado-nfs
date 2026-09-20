#ifndef CADO_LAS_GPU_ENGINE_HPP
#define CADO_LAS_GPU_ENGINE_HPP

#include <cstddef>
#include <cstdint>

#include "las-gpu-plan.hpp"

/* Backend-neutral interface to the two offloaded sieve stages.
 *
 * A "task" owns whatever the backend needs to keep around between calls
 * (device copies of the piecewise-linear approximation, of the
 * j-divisibility tables, scratch buffers, ...). Creating one is
 * expensive, calling it is not, so tasks are created once per special-q
 * and used for every bucket region of that special-q.
 */

struct gpu_norm_task;
struct gpu_survivor_task;

struct gpu_engine {
    virtual ~gpu_engine();

    virtual char const * name() const = 0;

    /* --- norm initialisation ------------------------------------- */

    /* segments[] is host memory; the backend copies what it needs. */
    virtual gpu_norm_task * norm_task_create(gpu_norm_plan const & plan,
                                             gpu_norm_segment const * segments,
                                             int nsegments) = 0;
    virtual void norm_task_destroy(gpu_norm_task * task) = 0;

    /* Fill nregions consecutive bucket regions, starting at region N0,
     * into S, which must hold nregions << log_bucket_region bytes.
     * Batching is what makes the device worth its transfer latency; a
     * single region is 64kB, which is nothing.
     */
    virtual void norm_fill(gpu_norm_task * task, unsigned char * S,
                           unsigned int N0, unsigned int nregions) = 0;

    /* --- survivor search ----------------------------------------- */

    virtual gpu_survivor_task *
    survivor_task_create(gpu_jdiv_entry const * jdiv, uint32_t jdiv_size,
                         size_t max_region_size) = 0;
    virtual void survivor_task_destroy(gpu_survivor_task * task) = 0;

    /* S0/S1 are host buffers of region_size bytes (S1 may be null when
     * only one side is sieved). Non-survivors are overwritten with 255
     * on the Sf side, exactly as search_survivors_in_line() does.
     *
     * Survivor offsets, relative to the start of the region, are
     * appended to out[] in increasing order. Returns the number of
     * survivors found, which may exceed out_capacity, in which case out
     * holds the first out_capacity of them and the caller must retry
     * with a bigger buffer.
     */
    virtual size_t survivor_search(gpu_survivor_task * task,
                                   unsigned char * S0, unsigned char * S1,
                                   size_t region_size,
                                   gpu_survivor_region const & region,
                                   uint32_t * out, size_t out_capacity) = 0;
};

/* Always available. */
gpu_engine * gpu_engine_cpu_create();

/* Returns nullptr when cado-nfs was built without CUDA, when no device
 * is visible, or when `device` is out of range. Never throws.
 */
gpu_engine * gpu_engine_cuda_create(int device);

/* True when a CUDA backend was compiled in at all (it may still fail to
 * find a device at run time).
 */
bool gpu_engine_cuda_compiled();

#endif /* CADO_LAS_GPU_ENGINE_HPP */
