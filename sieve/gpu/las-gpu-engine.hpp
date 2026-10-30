#ifndef CADO_LAS_GPU_ENGINE_HPP
#define CADO_LAS_GPU_ENGINE_HPP

#include <cstddef>
#include <cstdint>

#include "las-gpu-plan.hpp"
#include "las-gpu-fib-plan.hpp"

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
struct gpu_fib_task;

struct gpu_fib_pass;
struct gpu_fib_state;

/* What a bucket-sieving pass hands back. Every pointer is owned by the
 * pass and stays valid until fib_release().
 *
 * Everything is laid out bucket-major: entry b * nslices + s is about
 * the updates that slice s put in bucket b. That is the order the
 * updates themselves sit in, back to back, so the updates of slices
 * [s0, s1) in bucket b are the contiguous range
 *
 *      updates[offsets[b * nslices + s0] .. offsets[b * nslices + s1])
 *
 * which is what lets the caller spread the slices over several bucket
 * arrays without moving anything twice.
 */
struct gpu_fib_output {
    /* false when the backend ran out of memory, in which case nothing
     * else in here is meaningful */
    bool ok;

    uint32_t nslices;
    uint32_t n_buckets;

    /* n_buckets is the number of buckets of the pass, i.e. the
     * plan's chunk_count; bucket b here is chunk_begin + b in the
     * sieve area. n_buckets * nslices values: */
    uint32_t const * counts;
    /* exclusive prefix sums of counts[], plus the total at the end */
    uint32_t const * offsets;
    gpu_fib_update const * updates;

    /* Lattices the kernel declined to handle (see GPU_FIB_HOST), as
     * indices into the entry array, in no particular order. Reported
     * by the first chunk only (chunk_begin == 0): the host replays
     * them over the whole area. */
    uint32_t const * host_entries;
    uint64_t nhost;
};

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

    /* --- bucket sieving ------------------------------------------ */

    /* The factor base does not depend on the special-q -- only the
     * q-lattice does -- so the entry and slice arrays are uploaded once
     * per side and then reused for every special-q. That is what makes
     * this worth offloading at all: a pass costs one small upload and
     * one large download, with no setup in between.
     *
     * entries[] is the (entry, root) pairs of the bucket-sieved part,
     * flattened; slices[] cuts it into the runs that share a slice
     * index. Both are host memory and are copied.
     */
    virtual gpu_fib_task * fib_task_create(gpu_fib_entry const * entries,
                                           uint32_t nentries,
                                           gpu_fib_slice const * slices,
                                           uint32_t nslices) = 0;
    virtual void fib_task_destroy(gpu_fib_task * task) = 0;

    /* Sieve the whole part for one special-q.
     *
     * fib_task_create() and fib_launch() return nullptr, and
     * fib_wait() an output with ok == false, when the backend runs out
     * of memory: the device may be shared with other processes, and las
     * then carries on without it. fib_release() must still be called on
     * a pass whose wait failed.
     *
     * fib_launch() queues the pass and returns at once when the backend
     * is asynchronous, so that the caller can get on with whatever does
     * not depend on the updates. fib_wait() blocks until they are in
     * host memory; it must be called exactly once per pass, and the
     * output it returns stays valid until fib_release(). Several passes
     * may be in flight at the same time, from different threads.
     *
     * Updates within one (bucket, slice) pair come in no particular
     * order: the device scatters with atomics. Nothing downstream
     * depends on that order -- apply_one_bucket adds logp at each
     * position, and resieving looks at every update of a bucket region.
     */
    virtual gpu_fib_pass * fib_launch(gpu_fib_task * task,
                                      gpu_fib_plan const & plan,
                                      gpu_fib_state * state) = 0;

    /* Where each lattice's walk stopped, so that the next chunk of the
     * same special-q resumes from there instead of starting over.
     * Required when the area is done in several chunks, which must
     * then be launched in increasing order, each one after fib_wait()
     * on the previous; may be nullptr for a pass over the whole area.
     * One state serves one special-q at a time. nullptr when out of
     * memory. */
    virtual gpu_fib_state * fib_state_create(gpu_fib_task * task) = 0;
    virtual void fib_state_destroy(gpu_fib_state * state) = 0;
    virtual gpu_fib_output fib_wait(gpu_fib_pass * pass) = 0;
    virtual void fib_release(gpu_fib_pass * pass) = 0;
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
