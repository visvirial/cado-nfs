/* CUDA backend for the norm-initialisation and survivor-search stages.
 *
 * This translation unit is compiled by nvcc and, on purpose, includes no
 * cado-nfs header other than the three self-contained ones below. All
 * the arithmetic lives in las-gpu-kernels.hpp and is shared with the CPU
 * backend, so that the two cannot drift apart.
 *
 * Build it with -fmad=false: contracting a*b+c into an fma would change
 * the last bit of the lognorms and, with it, the set of survivors.
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include <algorithm>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
/* cub::CountingInputIterator is gone in cccl 3 (cuda 13); the thrust
 * spelling is the one that works with both. */
#include <thrust/iterator/counting_iterator.h>

#include "las-gpu-engine.hpp"
#include "las-gpu-fib-kernels.hpp"
#include "las-gpu-fib-plan.hpp"
#include "las-gpu-kernels.hpp"
#include "las-gpu-plan.hpp"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t const cado_cuda_err_ = (call);                            \
        if (cado_cuda_err_ != cudaSuccess) {                                  \
            fprintf(stderr, "%s:%d: CUDA error in %s: %s\n", __FILE__,        \
                    __LINE__, #call, cudaGetErrorString(cado_cuda_err_));     \
            abort();                                                          \
        }                                                                     \
    } while (0)

/* For the allocations of the bucket-sieving path, which las can recover
 * from: the device may well be shared with other processes, and run
 * out. Memory exhaustion is reported as a failure, which las answers by
 * going back to the CPU for a while; anything else is a bug and aborts
 * as above. The error is cleared either way, or the next
 * CUDA_CHECK(cudaGetLastError()) would trip over it.
 *
 * Within cuda_engine, CUDA_ALLOC first tries quietly, and on failure
 * trims the engine's pools and tries once more: the scratch of passes
 * that are not in flight is often what stands in the way. */
#define CUDA_ALLOC_ONCE(call, quiet)                                          \
    cuda_alloc_ok((call), #call, __FILE__, __LINE__, (quiet))
#define CUDA_ALLOC(call)                                                      \
    (CUDA_ALLOC_ONCE(call, true) || (trim(), CUDA_ALLOC_ONCE(call, false)))
static bool cuda_alloc_ok(cudaError_t err, char const * what,
                          char const * file, int line, bool quiet)
{
    if (err == cudaSuccess)
        return true;
    (void) cudaGetLastError();
    if (err == cudaErrorMemoryAllocation) {
        if (!quiet)
            fprintf(stderr, "# las-gpu: %s failed: %s\n", what,
                    cudaGetErrorString(err));
        return false;
    }
    fprintf(stderr, "%s:%d: CUDA error in %s: %s\n", file, line, what,
            cudaGetErrorString(err));
    abort();
}

#define CUDA_TRY(call)                                                        \
    do {                                                                      \
        cudaError_t const cado_cuda_err_ = (call);                            \
        if (cado_cuda_err_ != cudaSuccess) {                                  \
            cudaGetLastError();                                               \
            return false;                                                     \
        }                                                                     \
    } while (0)

namespace {

/* {{{ kernels */

__global__ void kernel_fill_norms(unsigned char * __restrict__ S,
                                  gpu_norm_plan plan, unsigned int N0,
                                  unsigned int nregions)
{
    uint32_t const region_size = 1u << plan.log_bucket_region;
    uint64_t const total = uint64_t(region_size) * nregions;
    uint64_t const stride = uint64_t(blockDim.x) * gridDim.x;

    for (uint64_t t = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         t < total; t += stride) {
        uint32_t const r = uint32_t(t >> plan.log_bucket_region);
        uint32_t const x = uint32_t(t) & (region_size - 1);
        auto const geo = gpu_region_geometry_for(plan.logI,
                                                 plan.log_bucket_region,
                                                 N0 + r);
        S[t] = gpu_norm_cell(plan, geo, x);
    }
}

/* One thread per cell: decide, write the 255 markers back into Sf, and
 * leave a flag behind for the compaction pass. */
__global__ void kernel_survivor_flags(unsigned char * __restrict__ S0,
                                      unsigned char * __restrict__ S1,
                                      uint8_t * __restrict__ flags,
                                      uint32_t region_size,
                                      gpu_survivor_region region,
                                      gpu_survivor_tables tables)
{
    uint32_t const stride = blockDim.x * gridDim.x;
    unsigned char * const Sf = region.side_of_Sf == 0 ? S0 : S1;

    for (uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; x < region_size;
         x += stride) {
        unsigned char const s0 = S0 ? S0[x] : 255;
        unsigned char const s1 = S1 ? S1[x] : 255;
        unsigned char sf_out;
        bool const alive =
            gpu_survivor_cell(region, tables, s0, s1, x, sf_out);
        Sf[x] = sf_out;
        flags[x] = alive ? 1 : 0;
    }
}

/* }}} */
/* {{{ bucket sieving kernels
 *
 * One thread per (factor base entry, root) pair. The walk is a serial
 * state machine, so there is nothing to parallelise inside one lattice;
 * the parallelism is that there are tens of thousands of them.
 *
 * The pass is done twice, once to count and once to write. That is the
 * price of keeping the updates grouped by slice in each bucket, which
 * is how apply_one_bucket recovers logp: the slice boundaries have to
 * be known before the first update can be placed. Re-walking is
 * cheaper than storing every position and sorting it.
 */

/* Where the hint of an entry comes from when the entries are shared
 * between the slicings of a factor base: it is the entry's index within
 * its slice, and every root of an entry is a lattice of its own, so the
 * hint of lattice k is (k - first) / nroots. See fib_task_create. */
struct cuda_fib_geom {
    uint32_t first;
    uint32_t nroots;
};

/* Where a lattice's walk starts in this pass: from scratch in the
 * first chunk, from where the previous chunk left it otherwise. */
static __device__ __forceinline__ void
fib_resume(gpu_ple & ple, gpu_fib_plan const & plan,
           uint64_t const * __restrict__ state, uint32_t k)
{
    if (state && plan.chunk_begin)
        ple.x = state[k];
}

__global__ void kernel_fib_count(gpu_fib_entry const * __restrict__ entries,
                                 uint32_t const * __restrict__ entry_slice,
                                 uint32_t nentries, gpu_fib_plan plan,
                                 uint32_t nslices,
                                 uint64_t const * __restrict__ state,
                                 uint32_t * __restrict__ counts,
                                 uint32_t * __restrict__ host_entries,
                                 uint32_t host_capacity,
                                 unsigned long long * __restrict__ nhost)
{
    gpu_fib_fence const F = gpu_fib_fence_for(plan.logI, plan.J);
    int const logB = plan.log_bucket_region;
    uint64_t const x_end = uint64_t(plan.chunk_begin + plan.chunk_count)
                           << logB;
    uint32_t const stride = blockDim.x * gridDim.x;

    for (uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < nentries;
         k += stride) {
        gpu_fib_entry const e = entries[k];
        if (!gpu_fib_entry_is_sieved(plan, e.p))
            continue;

        gpu_ple ple {};
        int const kind = gpu_fib_prepare(&ple, e, plan.basis, F, plan.logI);
        if (kind == GPU_FIB_HOST) {
            if (plan.chunk_begin == 0) {
                unsigned long long const slot = atomicAdd(nhost, 1ULL);
                if (slot < host_capacity)
                    host_entries[slot] = k;
            }
            continue;
        }
        fib_resume(ple, plan, state, k);

        uint32_t * const col = counts + entry_slice[k];
        gpu_fib_walk_until(ple, kind, F, x_end, [&](uint64_t x) {
            atomicAdd(&col[size_t((x >> logB) - plan.chunk_begin) * nslices],
                      1u);
        });
    }
}

/* cursor[] starts out as the exclusive prefix sums of counts[], and
 * offsets[grid] is the total. When that exceeds `capacity` the kernel
 * does nothing at all -- in particular it leaves the walk state alone
 * -- and the host grows the buffer and runs it again. */
__global__ void kernel_fib_scatter(gpu_fib_entry const * __restrict__ entries,
                                   uint32_t const * __restrict__ entry_slice,
                                   cuda_fib_geom const * __restrict__ geom,
                                   uint32_t nentries, gpu_fib_plan plan,
                                   uint32_t nslices,
                                   uint32_t const * __restrict__ total,
                                   uint64_t * __restrict__ state,
                                   uint32_t * __restrict__ cursor,
                                   gpu_fib_update * __restrict__ updates,
                                   uint32_t capacity)
{
    if (*total > capacity)
        return;

    gpu_fib_fence const F = gpu_fib_fence_for(plan.logI, plan.J);
    int const logB = plan.log_bucket_region;
    uint32_t const bmask = (UINT32_C(1) << logB) - 1;
    uint64_t const x_end = uint64_t(plan.chunk_begin + plan.chunk_count)
                           << logB;
    bool const last = plan.chunk_begin + plan.chunk_count >= plan.n_buckets;
    uint32_t const stride = blockDim.x * gridDim.x;

    for (uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < nentries;
         k += stride) {
        gpu_fib_entry const e = entries[k];
        if (!gpu_fib_entry_is_sieved(plan, e.p))
            continue;

        gpu_ple ple {};
        int const kind = gpu_fib_prepare(&ple, e, plan.basis, F, plan.logI);
        if (kind == GPU_FIB_HOST)
            continue;
        fib_resume(ple, plan, state, k);

        uint32_t const s = entry_slice[k];
        uint32_t * const col = cursor + s;
        uint16_t const hint = geom
            ? uint16_t((k - geom[s].first) / geom[s].nroots) : e.hint;
        uint64_t const resume = gpu_fib_walk_until(ple, kind, F, x_end,
                [&](uint64_t x) {
            uint32_t const slot = atomicAdd(
                    &col[size_t((x >> logB) - plan.chunk_begin) * nslices],
                    1u);
            updates[slot] = gpu_fib_update {hint, uint16_t(x & bmask)};
        });
        if (state && !last)
            state[k] = resume;
    }
}

/* }}} */
/* {{{ tasks */

/* Scratch for one in-flight norm fill. A task is shared by every
 * worker thread -- lognorm_gpu lives in nfs_work::side_data, and
 * process_bucket_region fills norms from all of them at once -- so the
 * buffer the kernel writes into cannot live in the task. Each call
 * takes a slot and gives it back; the pool grows to the number of
 * threads that ever ask at the same time.
 */
struct cuda_norm_slot {
    unsigned char * d_S = nullptr;
    unsigned char * h_S = nullptr;
    size_t bytes = 0;
    cudaStream_t stream = nullptr;
};

struct cuda_norm_task {
    gpu_norm_plan plan;          /* .segments points to device memory */
    gpu_norm_segment * d_segments = nullptr;

    std::mutex mu;
    std::vector<cuda_norm_slot *> free_slots;
    std::vector<cuda_norm_slot *> all_slots;
};

/* How many survivors we fetch without being asked twice. A 64kB region
 * yields a few hundred at the densities las actually sieves at, so this
 * covers it with room to spare; the slow path below catches the rest.
 */
#define CUDA_SURVIVOR_PREFETCH 4096

/* see fib_launch */
static constexpr uint32_t CUDA_FIB_HOST_MAX = UINT32_C(1) << 20;

/* The device copy of a flattened factor base, minus its hints. las has
 * one task per slicing, and several slicings per side that differ only
 * in where the slices start and end; at c270 sizes, a copy for each is
 * gigabytes. So the tasks share the entries when they can, and keep to
 * themselves only which slice each entry is in. Owned by the engine,
 * which counts the tasks that use it. */
struct cuda_fib_entries {
    gpu_fib_entry * d = nullptr;
    uint32_t n = 0;
    uint64_t hash = 0;
    unsigned refs = 0;
};

struct cuda_fib_task {
    /* uploaded once, reused for every special-q: the factor base does
     * not depend on q */
    gpu_fib_entry const * d_entries = nullptr;
    uint32_t * d_entry_slice = nullptr;
    /* with shared entries, whence the hints; nullptr when the entries
     * are this task's own, hints included */
    cuda_fib_geom * d_geom = nullptr;
    cuda_fib_entries * shared = nullptr;
    gpu_fib_entry * d_own = nullptr;
    uint32_t nentries = 0;
    uint32_t nslices = 0;
};

/* Where every lattice's walk stopped, for the next chunk. The chunks
 * of a special-q go through different streams, so the event orders
 * each one's count after the previous one's scatter. */
struct cuda_fib_state {
    uint64_t * d_x = nullptr;
    cudaEvent_t done = nullptr;
};

/* Scratch for one in-flight bucket-sieving pass, which is also what a
 * gpu_fib_pass points to.
 *
 * This is kept apart from cuda_fib_task because there is one task per
 * factor base slicing, and las uses several of those per side -- the
 * slicing is keyed on the lognorm scale, which moves with the
 * special-q. The scratch, on the other hand, is sized by the sieve
 * area and is mostly page-locked memory, which costs milliseconds to
 * allocate. Pooling it in the engine means the first few special-q's
 * pay for it and nobody else does. A slot is taken by fib_launch and
 * given back by fib_release; the pool grows to the number of passes
 * ever in flight at once.
 */
struct cuda_fib_scratch {
    /* Device side. counts[] has one extra, always zero, element so that
     * the scan writes the total at offsets[grid]. */
    uint32_t * d_counts = nullptr;
    uint32_t * d_offsets = nullptr;
    uint32_t * d_cursor = nullptr;
    size_t grid_capacity = 0;       /* in nslices*nb units */
    void * d_scan_temp = nullptr;
    size_t scan_temp_bytes = 0;

    gpu_fib_update * d_updates = nullptr;
    uint32_t updates_capacity = 0;

    uint32_t * d_host_entries = nullptr;
    unsigned long long * d_nhost = nullptr;
    uint32_t host_capacity = 0;

    /* Page-locked mirrors, so that every copy can be queued. The
     * bucket arrays point straight into h_updates rather than getting
     * a copy, which is why a pass keeps its scratch until it is
     * released.
     *
     * h_updates, by far the largest, is not owned by the scratch: it is
     * taken from the engine's pool by fib_wait and given back by
     * fib_release. A pass that has been launched but not waited for --
     * one prefetched for the next special-q -- therefore holds none,
     * which halves the page-locked memory las needs. */
    uint32_t * h_counts = nullptr;
    uint32_t * h_offsets = nullptr;
    gpu_fib_update * h_updates = nullptr;
    uint32_t h_updates_capacity = 0;
    uint32_t * h_host_entries = nullptr;
    unsigned long long * h_nhost = nullptr;

    cudaStream_t stream = nullptr;

    /* the pass in flight */
    cuda_fib_task const * task = nullptr;
    cuda_fib_state * state = nullptr;
    gpu_fib_plan plan;
    int grid_e = 0;
};

struct cuda_survivor_task {
    gpu_jdiv_entry * d_jdiv = nullptr;
    uint32_t jdiv_size = 0;

    /* One allocation for both sides, so that the two sieve arrays go up
     * in a single transfer. d_S1 is just d_S0 + capacity. */
    unsigned char * d_S0 = nullptr;
    unsigned char * d_S1 = nullptr;
    uint8_t * d_flags = nullptr;
    uint32_t * d_out = nullptr;
    uint32_t * d_num = nullptr;
    void * d_temp = nullptr;
    size_t d_temp_bytes = 0;
    size_t capacity = 0;

    /* Everything below exists so that one call costs one synchronisation.
     * Staging through page-locked memory is what lets the copies be
     * issued asynchronously; a transfer from pageable memory would be
     * staged by the driver anyway, and would block.
     */
    cudaStream_t stream = nullptr;
    unsigned char * h_stage = nullptr;  /* 2*capacity, upload */
    unsigned char * h_Sf = nullptr;     /* capacity, download */
    uint32_t * h_out = nullptr;         /* CUDA_SURVIVOR_PREFETCH */
    uint32_t * h_num = nullptr;         /* 1 */
};

/* }}} */

struct cuda_engine : public gpu_engine {
    int device;
    /* large enough for the longest prop.name plus the "cuda:%d ()" frame */
    char label[sizeof(cudaDeviceProp::name) + 32];

    explicit cuda_engine(int device)
        : device(device)
    {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device) == cudaSuccess)
            snprintf(label, sizeof(label), "cuda:%d (%s)", device, prop.name);
        else
            snprintf(label, sizeof(label), "cuda:%d", device);
    }

    char const * name() const override { return label; }

    /* --- norms --------------------------------------------------- */

    gpu_norm_task * norm_task_create(gpu_norm_plan const & plan,
                                     gpu_norm_segment const * segments,
                                     int nsegments) override
    {
        CUDA_CHECK(cudaSetDevice(device));
        auto * t = new cuda_norm_task;
        t->plan = plan;
        t->plan.nsegments = nsegments;
        if (nsegments > 0) {
            size_t const bytes = size_t(nsegments) * sizeof(gpu_norm_segment);
            CUDA_CHECK(cudaMalloc(&t->d_segments, bytes));
            CUDA_CHECK(cudaMemcpy(t->d_segments, segments, bytes,
                                  cudaMemcpyHostToDevice));
        }
        t->plan.segments = t->d_segments;
        return reinterpret_cast<gpu_norm_task *>(t);
    }

    void norm_task_destroy(gpu_norm_task * task) override
    {
        auto * t = reinterpret_cast<cuda_norm_task *>(task);
        if (!t)
            return;
        cudaSetDevice(device);
        cudaFree(t->d_segments);
        for (auto * sl: t->all_slots) {
            cudaFree(sl->d_S);
            cudaFreeHost(sl->h_S);
            if (sl->stream)
                cudaStreamDestroy(sl->stream);
            delete sl;
        }
        delete t;
    }

    void norm_fill(gpu_norm_task * task, unsigned char * S, unsigned int N0,
                   unsigned int nregions) override
    {
        auto * t = reinterpret_cast<cuda_norm_task *>(task);
        CUDA_CHECK(cudaSetDevice(device));

        size_t const bytes = (size_t(nregions) << t->plan.log_bucket_region);

        cuda_norm_slot * sl = nullptr;
        {
            std::lock_guard<std::mutex> const lock(t->mu);
            if (!t->free_slots.empty()) {
                sl = t->free_slots.back();
                t->free_slots.pop_back();
            } else {
                sl = new cuda_norm_slot;
                CUDA_CHECK(cudaStreamCreate(&sl->stream));
                t->all_slots.push_back(sl);
            }
        }

        if (bytes > sl->bytes) {
            cudaFree(sl->d_S);
            cudaFreeHost(sl->h_S);
            CUDA_CHECK(cudaMalloc(&sl->d_S, bytes));
            CUDA_CHECK(cudaHostAlloc(&sl->h_S, bytes, cudaHostAllocDefault));
            sl->bytes = bytes;
        }

        int const block = 256;
        int grid = int((bytes + block - 1) / block);
        if (grid > 65535)
            grid = 65535;

        kernel_fill_norms<<<grid, block, 0, sl->stream>>>(sl->d_S, t->plan, N0,
                                                          nregions);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(sl->h_S, sl->d_S, bytes,
                                   cudaMemcpyDeviceToHost, sl->stream));
        CUDA_CHECK(cudaStreamSynchronize(sl->stream));
        memcpy(S, sl->h_S, bytes);

        {
            std::lock_guard<std::mutex> const lock(t->mu);
            t->free_slots.push_back(sl);
        }
    }

    /* --- survivors ----------------------------------------------- */

    gpu_survivor_task * survivor_task_create(gpu_jdiv_entry const * jdiv,
                                             uint32_t jdiv_size,
                                             size_t max_region_size) override
    {
        CUDA_CHECK(cudaSetDevice(device));
        auto * t = new cuda_survivor_task;
        t->jdiv_size = jdiv_size;
        size_t const jbytes = size_t(jdiv_size) * sizeof(gpu_jdiv_entry);
        CUDA_CHECK(cudaMalloc(&t->d_jdiv, jbytes));
        CUDA_CHECK(cudaMemcpy(t->d_jdiv, jdiv, jbytes, cudaMemcpyHostToDevice));

        t->capacity = max_region_size;
        CUDA_CHECK(cudaMalloc(&t->d_S0, 2 * max_region_size));
        t->d_S1 = t->d_S0 + max_region_size;
        CUDA_CHECK(cudaMalloc(&t->d_flags, max_region_size));
        CUDA_CHECK(cudaMalloc(&t->d_out, max_region_size * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&t->d_num, sizeof(uint32_t)));

        CUDA_CHECK(cudaStreamCreate(&t->stream));
        CUDA_CHECK(cudaHostAlloc(&t->h_stage, 2 * max_region_size,
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&t->h_Sf, max_region_size,
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&t->h_out,
                                 CUDA_SURVIVOR_PREFETCH * sizeof(uint32_t),
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&t->h_num, sizeof(uint32_t),
                                 cudaHostAllocDefault));

        /* cub wants to be told how much scratch it needs. */
        thrust::counting_iterator<uint32_t> const indices(0);
        CUDA_CHECK(cub::DeviceSelect::Flagged(
            nullptr, t->d_temp_bytes, indices, t->d_flags, t->d_out, t->d_num,
            int(max_region_size)));
        CUDA_CHECK(cudaMalloc(&t->d_temp, t->d_temp_bytes));

        return reinterpret_cast<gpu_survivor_task *>(t);
    }

    void survivor_task_destroy(gpu_survivor_task * task) override
    {
        auto * t = reinterpret_cast<cuda_survivor_task *>(task);
        if (!t)
            return;
        cudaSetDevice(device);
        cudaFree(t->d_jdiv);
        cudaFree(t->d_S0);      /* d_S1 points inside this one */
        cudaFree(t->d_flags);
        cudaFree(t->d_out);
        cudaFree(t->d_num);
        cudaFree(t->d_temp);
        cudaFreeHost(t->h_stage);
        cudaFreeHost(t->h_Sf);
        cudaFreeHost(t->h_out);
        cudaFreeHost(t->h_num);
        if (t->stream)
            cudaStreamDestroy(t->stream);
        delete t;
    }

    size_t survivor_search(gpu_survivor_task * task, unsigned char * S0,
                           unsigned char * S1, size_t region_size,
                           gpu_survivor_region const & region, uint32_t * out,
                           size_t out_capacity) override
    {
        auto * t = reinterpret_cast<cuda_survivor_task *>(task);
        CUDA_CHECK(cudaSetDevice(device));

        if (region_size > t->capacity) {
            fprintf(stderr,
                    "las-gpu: survivor task built for %zu bytes, asked for "
                    "%zu\n",
                    t->capacity, region_size);
            abort();
        }

        /* Stage both sides into one page-locked block and send them up
         * together: two 64kB transfers cost appreciably more than one
         * of 128kB, and the whole point here is the per-call overhead,
         * not the bandwidth. */
        cudaStream_t const stream = t->stream;
        if (S0 && S1) {
            /* contiguous on both ends, so this is a single transfer */
            memcpy(t->h_stage, S0, region_size);
            memcpy(t->h_stage + t->capacity, S1, region_size);
            CUDA_CHECK(cudaMemcpyAsync(t->d_S0, t->h_stage,
                                       t->capacity + region_size,
                                       cudaMemcpyHostToDevice, stream));
        } else if (S0 || S1) {
            unsigned char const * const src = S0 ? S0 : S1;
            unsigned char * const dst = S0 ? t->d_S0 : t->d_S1;
            unsigned char * const stage =
                S0 ? t->h_stage : t->h_stage + t->capacity;
            memcpy(stage, src, region_size);
            CUDA_CHECK(cudaMemcpyAsync(dst, stage, region_size,
                                       cudaMemcpyHostToDevice, stream));
        }

        gpu_survivor_tables tables {};
        tables.jdiv = t->d_jdiv;
        tables.jdiv_size = t->jdiv_size;

        int const block = 256;
        int grid = int((region_size + block - 1) / block);
        if (grid > 65535)
            grid = 65535;

        kernel_survivor_flags<<<grid, block, 0, stream>>>(
            S0 ? t->d_S0 : nullptr, S1 ? t->d_S1 : nullptr, t->d_flags,
            uint32_t(region_size), region, tables);
        CUDA_CHECK(cudaGetLastError());

        /* DeviceSelect::Flagged keeps the input order, which is exactly
         * the "increasing x" contract of search_survivors(). */
        thrust::counting_iterator<uint32_t> const indices(0);
        CUDA_CHECK(cub::DeviceSelect::Flagged(t->d_temp, t->d_temp_bytes,
                                              indices, t->d_flags, t->d_out,
                                              t->d_num, int(region_size),
                                              stream));

        /* Only the Sf side is modified by the kernel. */
        unsigned char * const Sf = region.side_of_Sf == 0 ? S0 : S1;
        unsigned char const * const d_Sf =
            region.side_of_Sf == 0 ? t->d_S0 : t->d_S1;

        /* Queue the count, the markers and a first helping of survivors
         * in one go. Reading the count first, as the obvious version
         * does, costs a full round trip before the other two can even be
         * issued -- three synchronisations per bucket region instead of
         * one, which at this granularity is most of the cost. */
        size_t const prefetch = out_capacity < CUDA_SURVIVOR_PREFETCH
                              ? out_capacity : CUDA_SURVIVOR_PREFETCH;
        CUDA_CHECK(cudaMemcpyAsync(t->h_num, t->d_num, sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(t->h_Sf, d_Sf, region_size,
                                   cudaMemcpyDeviceToHost, stream));
        if (prefetch)
            CUDA_CHECK(cudaMemcpyAsync(t->h_out, t->d_out,
                                       prefetch * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        uint32_t const n = *t->h_num;
        memcpy(Sf, t->h_Sf, region_size);

        size_t const ncopy = n < out_capacity ? n : out_capacity;
        if (ncopy <= prefetch) {
            memcpy(out, t->h_out, ncopy * sizeof(uint32_t));
        } else {
            /* Rare enough that a second round trip does not matter. */
            CUDA_CHECK(cudaMemcpy(out, t->d_out, ncopy * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
        }
        return n;
    }

    /* --- bucket sieving ------------------------------------------ */

    std::mutex fib_mu;
    std::vector<cuda_fib_scratch *> fib_free;
    std::vector<cuda_fib_scratch *> fib_all;

    /* page-locked landing zones for the updates, see cuda_fib_scratch */
    struct host_buffer {
        gpu_fib_update * p;
        uint32_t capacity;
    };
    std::vector<host_buffer> fib_host_free;

    bool fib_host_take(cuda_fib_scratch * sc, uint32_t total)
    {
        host_buffer hb {nullptr, 0};
        {
            std::lock_guard<std::mutex> const lock(fib_mu);
            /* the smallest that fits, or else the largest, to grow */
            auto best = fib_host_free.end();
            for (auto it = fib_host_free.begin(); it != fib_host_free.end(); ++it) {
                if (best == fib_host_free.end())
                    best = it;
                else if (it->capacity >= total &&
                        (best->capacity < total || it->capacity < best->capacity))
                    best = it;
                else if (best->capacity < total && it->capacity > best->capacity)
                    best = it;
            }
            if (best != fib_host_free.end()) {
                hb = *best;
                fib_host_free.erase(best);
            }
        }
        if (hb.capacity < total) {
            /* Grow generously: the high-water mark drifts up over the
             * first few special-q's and allocating page-locked memory
             * is not cheap. */
            cudaFreeHost(hb.p);
            hb = host_buffer {nullptr, 0};
            uint64_t want = uint64_t(total) + total / 4;
            if (want > UINT32_MAX)
                want = UINT32_MAX;
            if (!CUDA_ALLOC(cudaHostAlloc(&hb.p,
                                          size_t(want) *
                                              sizeof(gpu_fib_update),
                                          cudaHostAllocDefault)))
                return false;
            hb.capacity = uint32_t(want);
        }
        sc->h_updates = hb.p;
        sc->h_updates_capacity = hb.capacity;
        return true;
    }

    void fib_host_give(cuda_fib_scratch * sc)
    {
        if (!sc->h_updates)
            return;
        std::lock_guard<std::mutex> const lock(fib_mu);
        fib_host_free.push_back(host_buffer {sc->h_updates,
                                             sc->h_updates_capacity});
        sc->h_updates = nullptr;
        sc->h_updates_capacity = 0;
    }

    size_t available_memory() override
    {
        size_t free = 0, total = 0;
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        return free;
    }

    /* Only the scratch nobody holds: a slot in fib_free has no pass in
     * flight (fib_release synchronised its stream), and the lock keeps
     * fib_scratch_take from handing it out meanwhile. The capacities go
     * to zero, so that the next fib_launch allocates afresh. */
    void trim() override
    {
        cudaSetDevice(device);
        std::lock_guard<std::mutex> const lock(fib_mu);
        for (auto * sc: fib_free) {
            cudaFree(sc->d_counts);
            cudaFree(sc->d_offsets);
            cudaFree(sc->d_cursor);
            cudaFree(sc->d_scan_temp);
            cudaFree(sc->d_updates);
            cudaFree(sc->d_host_entries);
            cudaFreeHost(sc->h_counts);
            cudaFreeHost(sc->h_offsets);
            cudaFreeHost(sc->h_host_entries);
            sc->d_counts = sc->d_offsets = sc->d_cursor = nullptr;
            sc->d_scan_temp = nullptr;
            sc->d_updates = nullptr;
            sc->d_host_entries = nullptr;
            sc->h_counts = sc->h_offsets = sc->h_host_entries = nullptr;
            sc->grid_capacity = 0;
            sc->scan_temp_bytes = 0;
            sc->updates_capacity = 0;
            sc->host_capacity = 0;
        }
        for (auto const & hb: fib_host_free)
            cudaFreeHost(hb.p);
        fib_host_free.clear();
    }

    ~cuda_engine() override
    {
        cudaSetDevice(device);
        for (auto * sc: fib_all) {
            cudaFree(sc->d_counts);
            cudaFree(sc->d_offsets);
            cudaFree(sc->d_cursor);
            cudaFree(sc->d_scan_temp);
            cudaFree(sc->d_updates);
            cudaFree(sc->d_host_entries);
            cudaFree(sc->d_nhost);
            cudaFreeHost(sc->h_counts);
            cudaFreeHost(sc->h_offsets);
            cudaFreeHost(sc->h_updates);
            cudaFreeHost(sc->h_host_entries);
            cudaFreeHost(sc->h_nhost);
            if (sc->stream)
                cudaStreamDestroy(sc->stream);
            delete sc;
        }
        for (auto const & hb: fib_host_free)
            cudaFreeHost(hb.p);
    }

    cuda_fib_scratch * fib_scratch_take()
    {
        {
            std::lock_guard<std::mutex> const lock(fib_mu);
            if (!fib_free.empty()) {
                auto * sc = fib_free.back();
                fib_free.pop_back();
                return sc;
            }
        }
        auto * sc = new cuda_fib_scratch;
        if (!CUDA_ALLOC(cudaStreamCreate(&sc->stream)) ||
                !CUDA_ALLOC(cudaMalloc(&sc->d_nhost,
                                       sizeof(unsigned long long))) ||
                !CUDA_ALLOC(cudaHostAlloc(&sc->h_nhost,
                                          sizeof(unsigned long long),
                                          cudaHostAllocDefault))) {
            cudaFree(sc->d_nhost);
            if (sc->stream)
                cudaStreamDestroy(sc->stream);
            delete sc;
            return nullptr;
        }
        std::lock_guard<std::mutex> const lock(fib_mu);
        fib_all.push_back(sc);
        return sc;
    }

    std::mutex fib_entries_mu;
    std::vector<cuda_fib_entries *> fib_entries;

    /* Of what the kernels read from an entry other than the hint. */
    static uint64_t fib_entries_hash(gpu_fib_entry const * entries,
                                     uint32_t n)
    {
        uint64_t h = 0x9e3779b97f4a7c15 ^ n;
        for (uint32_t k = 0; k < n; k++) {
            uint64_t v = (uint64_t(entries[k].p) << 32 | entries[k].invq) ^
                         (uint64_t(entries[k].r) * 0xff51afd7ed558ccd);
            v ^= v >> 33;
            v *= 0xc4ceb9fe1a85ec53;
            h = (h ^ v) * 0x100000001b3 + k;
        }
        return h;
    }

    /* A hash match is checked in full against the device copy, a piece
     * at a time: a wrong match would lose relations without a word. */
    bool fib_entries_equal(cuda_fib_entries const * E,
                           gpu_fib_entry const * entries)
    {
        constexpr uint32_t piece = 1u << 20;
        std::vector<gpu_fib_entry> buf(std::min(E->n, piece));
        for (uint32_t k0 = 0; k0 < E->n; k0 += piece) {
            uint32_t const m = std::min(piece, E->n - k0);
            CUDA_CHECK(cudaMemcpy(buf.data(), E->d + k0,
                                  size_t(m) * sizeof(gpu_fib_entry),
                                  cudaMemcpyDeviceToHost));
            for (uint32_t k = 0; k < m; k++) {
                gpu_fib_entry const & a = buf[k];
                gpu_fib_entry const & b = entries[k0 + k];
                if (a.p != b.p || a.invq != b.invq || a.r != b.r)
                    return false;
            }
        }
        return true;
    }

    /* The device copy of these entries, shared with the tasks that
     * already have the same ones, or uploaded. nullptr when out of
     * memory. Called with fib_entries_mu held. */
    cuda_fib_entries * fib_entries_take(gpu_fib_entry const * entries,
                                        uint32_t n)
    {
        uint64_t const h = fib_entries_hash(entries, n);
        for (auto * E: fib_entries) {
            if (E->n == n && E->hash == h && fib_entries_equal(E, entries)) {
                E->refs++;
                return E;
            }
        }
        auto * E = new cuda_fib_entries;
        size_t const bytes = size_t(n) * sizeof(gpu_fib_entry);
        if (!CUDA_ALLOC(cudaMalloc(&E->d, bytes))) {
            delete E;
            return nullptr;
        }
        CUDA_CHECK(cudaMemcpy(E->d, entries, bytes, cudaMemcpyHostToDevice));
        E->n = n;
        E->hash = h;
        E->refs = 1;
        fib_entries.push_back(E);
        return E;
    }

    void fib_entries_give(cuda_fib_entries * E)
    {
        std::lock_guard<std::mutex> const lock(fib_entries_mu);
        if (--E->refs)
            return;
        cudaFree(E->d);
        fib_entries.erase(std::find(fib_entries.begin(), fib_entries.end(),
                                    E));
        delete E;
    }

    gpu_fib_task * fib_task_create(gpu_fib_entry const * entries,
                                   uint32_t nentries,
                                   gpu_fib_slice const * slices,
                                   uint32_t nslices) override
    {
        CUDA_CHECK(cudaSetDevice(device));
        auto * t = new cuda_fib_task;
        t->nentries = nentries;
        t->nslices = nslices;

        /* The kernels want the slice of an entry, not the entries of a
         * slice: one thread is one entry. Flatten it here, once. */
        std::vector<uint32_t> es(nentries);
        for (uint32_t s = 0; s < nslices; s++)
            for (uint32_t k = 0; k < slices[s].count; k++)
                es[slices[s].first + k] = s;

        /* The entries can be shared when the hints follow from the
         * slices, which is how las lays them out: an entry's roots one
         * after the other, numbered from 0 in each slice. */
        std::vector<cuda_fib_geom> geom(nslices);
        bool derivable = true;
        for (uint32_t s = 0; s < nslices && derivable; s++) {
            gpu_fib_slice const & sl = slices[s];
            geom[s].first = sl.first;
            geom[s].nroots = 1;
            if (!sl.count)
                continue;
            uint32_t const nh =
                uint32_t(entries[sl.first + sl.count - 1].hint) + 1;
            if (sl.count % nh) {
                derivable = false;
                break;
            }
            geom[s].nroots = sl.count / nh;
            for (uint32_t k = 0; k < sl.count && derivable; k++)
                derivable = entries[sl.first + k].hint ==
                    uint16_t(k / geom[s].nroots);
        }

        auto fail = [&]() {
            cudaFree(t->d_entry_slice);
            cudaFree(t->d_geom);
            cudaFree(t->d_own);
            if (t->shared)
                fib_entries_give(t->shared);
            delete t;
            return nullptr;
        };

        if (!CUDA_ALLOC(cudaMalloc(&t->d_entry_slice,
                                   size_t(nentries) * sizeof(uint32_t))))
            return fail();
        CUDA_CHECK(cudaMemcpy(t->d_entry_slice, es.data(),
                              size_t(nentries) * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        if (derivable) {
            if (!CUDA_ALLOC(cudaMalloc(&t->d_geom,
                                       std::max<size_t>(1, nslices) *
                                           sizeof(cuda_fib_geom))))
                return fail();
            CUDA_CHECK(cudaMemcpy(t->d_geom, geom.data(),
                                  size_t(nslices) * sizeof(cuda_fib_geom),
                                  cudaMemcpyHostToDevice));
            {
                std::lock_guard<std::mutex> const lock(fib_entries_mu);
                t->shared = fib_entries_take(entries, nentries);
            }
            if (!t->shared)
                return fail();
            t->d_entries = t->shared->d;
        } else {
            size_t const ebytes = size_t(nentries) * sizeof(gpu_fib_entry);
            if (!CUDA_ALLOC(cudaMalloc(&t->d_own, ebytes)))
                return fail();
            CUDA_CHECK(cudaMemcpy(t->d_own, entries, ebytes,
                                  cudaMemcpyHostToDevice));
            t->d_entries = t->d_own;
        }
        return reinterpret_cast<gpu_fib_task *>(t);
    }

    void fib_task_destroy(gpu_fib_task * task) override
    {
        auto * t = reinterpret_cast<cuda_fib_task *>(task);
        if (!t)
            return;
        cudaSetDevice(device);
        cudaFree(t->d_entry_slice);
        cudaFree(t->d_geom);
        cudaFree(t->d_own);
        if (t->shared)
            fib_entries_give(t->shared);
        delete t;
    }

    static void fib_enqueue_scatter(cuda_fib_scratch * sc, size_t grid)
    {
        cuda_fib_task const * const t = sc->task;
        CUDA_CHECK(cudaMemcpyAsync(sc->d_cursor, sc->d_offsets,
                                   grid * sizeof(uint32_t),
                                   cudaMemcpyDeviceToDevice, sc->stream));
        kernel_fib_scatter<<<sc->grid_e, 128, 0, sc->stream>>>(
            t->d_entries, t->d_entry_slice, t->d_geom, t->nentries, sc->plan,
            t->nslices, sc->d_offsets + grid,
            sc->state ? sc->state->d_x : nullptr, sc->d_cursor,
            sc->d_updates, sc->updates_capacity);
        CUDA_CHECK(cudaGetLastError());
        /* the next chunk's count must see this scatter's walk state */
        if (sc->state)
            CUDA_CHECK(cudaEventRecord(sc->state->done, sc->stream));
    }

    gpu_fib_state * fib_state_create(gpu_fib_task * task) override
    {
        auto * t = reinterpret_cast<cuda_fib_task *>(task);
        CUDA_CHECK(cudaSetDevice(device));
        auto * st = new cuda_fib_state;
        if (!CUDA_ALLOC(cudaMalloc(&st->d_x,
                                   size_t(t->nentries) * sizeof(uint64_t)))) {
            delete st;
            return nullptr;
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&st->done,
                                            cudaEventDisableTiming));
        return reinterpret_cast<gpu_fib_state *>(st);
    }

    void fib_state_destroy(gpu_fib_state * state) override
    {
        auto * st = reinterpret_cast<cuda_fib_state *>(state);
        if (!st)
            return;
        cudaSetDevice(device);
        cudaFree(st->d_x);
        cudaEventDestroy(st->done);
        delete st;
    }

    /* Everything up to and including the scatter goes into the stream
     * here, with no synchronisation: the device needs nothing from the
     * host once it has the plan. Only copying the updates back has to
     * wait for fib_wait(), because their number is not known until the
     * count is done. */
    gpu_fib_pass * fib_launch(gpu_fib_task * task,
                              gpu_fib_plan const & plan,
                              gpu_fib_state * state) override
    {
        auto * t = reinterpret_cast<cuda_fib_task *>(task);
        CUDA_CHECK(cudaSetDevice(device));
        cuda_fib_scratch * const sc = fib_scratch_take();
        if (!sc)
            return nullptr;
        cudaStream_t const stream = sc->stream;
        sc->task = t;
        sc->state = reinterpret_cast<cuda_fib_state *>(state);
        sc->plan = plan;

        size_t const grid = size_t(t->nslices) * plan.chunk_count;

        if (grid + 1 > sc->grid_capacity) {
            cudaFree(sc->d_counts);
            cudaFree(sc->d_offsets);
            cudaFree(sc->d_cursor);
            cudaFree(sc->d_scan_temp);
            cudaFreeHost(sc->h_counts);
            cudaFreeHost(sc->h_offsets);
            sc->d_counts = sc->d_offsets = sc->d_cursor = nullptr;
            sc->d_scan_temp = nullptr;
            sc->h_counts = sc->h_offsets = nullptr;
            sc->grid_capacity = 0;
            size_t const n = grid + 1;
            sc->scan_temp_bytes = 0;
            CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                nullptr, sc->scan_temp_bytes, sc->d_counts, sc->d_offsets,
                n));
            if (!CUDA_ALLOC(cudaMalloc(&sc->d_counts, n * sizeof(uint32_t))) ||
                    !CUDA_ALLOC(cudaMalloc(&sc->d_offsets,
                                           n * sizeof(uint32_t))) ||
                    !CUDA_ALLOC(cudaMalloc(&sc->d_cursor,
                                           n * sizeof(uint32_t))) ||
                    !CUDA_ALLOC(cudaHostAlloc(&sc->h_counts,
                                              n * sizeof(uint32_t),
                                              cudaHostAllocDefault)) ||
                    !CUDA_ALLOC(cudaHostAlloc(&sc->h_offsets,
                                              n * sizeof(uint32_t),
                                              cudaHostAllocDefault)) ||
                    !CUDA_ALLOC(cudaMalloc(&sc->d_scan_temp,
                                           sc->scan_temp_bytes))) {
                /* what did get allocated stays, with grid_capacity 0
                 * so that the next launch starts over */
                fib_release(reinterpret_cast<gpu_fib_pass *>(sc));
                return nullptr;
            }
            sc->grid_capacity = n;
        }
        /* Lattices handed back to the host are rare -- projective-like
         * ones, a handful per special-q -- so a fixed buffer does; one
         * sized by the factor base would be gigabytes at c270 sizes. A
         * pass that has more is declined in fib_wait. */
        uint32_t const host_want = std::min(t->nentries, CUDA_FIB_HOST_MAX);
        if (host_want > sc->host_capacity) {
            cudaFree(sc->d_host_entries);
            cudaFreeHost(sc->h_host_entries);
            sc->d_host_entries = nullptr;
            sc->h_host_entries = nullptr;
            sc->host_capacity = 0;
            if (!CUDA_ALLOC(cudaMalloc(&sc->d_host_entries,
                                       size_t(host_want) *
                                           sizeof(uint32_t))) ||
                    !CUDA_ALLOC(cudaHostAlloc(&sc->h_host_entries,
                                              size_t(host_want) *
                                                  sizeof(uint32_t),
                                              cudaHostAllocDefault))) {
                fib_release(reinterpret_cast<gpu_fib_pass *>(sc));
                return nullptr;
            }
            sc->host_capacity = host_want;
        }

        CUDA_CHECK(cudaMemsetAsync(sc->d_counts, 0,
                                   (grid + 1) * sizeof(uint32_t), stream));
        CUDA_CHECK(cudaMemsetAsync(sc->d_nhost, 0, sizeof(unsigned long long),
                                   stream));

        sc->grid_e = int((t->nentries + 127) / 128);
        if (sc->grid_e > 65535)
            sc->grid_e = 65535;
        if (sc->grid_e == 0)
            sc->grid_e = 1;

        if (sc->state && plan.chunk_begin)
            CUDA_CHECK(cudaStreamWaitEvent(stream, sc->state->done, 0));

        kernel_fib_count<<<sc->grid_e, 128, 0, stream>>>(
            t->d_entries, t->d_entry_slice, t->nentries, plan, t->nslices,
            sc->state ? sc->state->d_x : nullptr,
            sc->d_counts, sc->d_host_entries, sc->host_capacity,
            sc->d_nhost);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            sc->d_scan_temp, sc->scan_temp_bytes, sc->d_counts, sc->d_offsets,
            grid + 1, stream));

        CUDA_CHECK(cudaMemcpyAsync(sc->h_counts, sc->d_counts,
                                   grid * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(sc->h_offsets, sc->d_offsets,
                                   (grid + 1) * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(sc->h_nhost, sc->d_nhost,
                                   sizeof(unsigned long long),
                                   cudaMemcpyDeviceToHost, stream));

        fib_enqueue_scatter(sc, grid);

        return reinterpret_cast<gpu_fib_pass *>(sc);
    }

    gpu_fib_output fib_wait(gpu_fib_pass * pass) override
    {
        auto * sc = reinterpret_cast<cuda_fib_scratch *>(pass);
        cuda_fib_task const * const t = sc->task;
        CUDA_CHECK(cudaSetDevice(device));
        cudaStream_t const stream = sc->stream;
        size_t const grid = size_t(t->nslices) * sc->plan.chunk_count;

        CUDA_CHECK(cudaStreamSynchronize(stream));
        uint32_t const total = sc->h_offsets[grid];
        uint64_t const nhost = *sc->h_nhost;

        if (nhost > sc->host_capacity) {
            fprintf(stderr, "# las-gpu: %llu lattices handed back to the"
                    " host, more than the %u expected\n",
                    (unsigned long long) nhost, sc->host_capacity);
            return gpu_fib_output {};
        }

        /* The scan runs in 32 bits. The host side never asks for a pass
         * that could get near that, but if one ever did, the offsets
         * would wrap silently and the updates overwrite each other --
         * losing relations without a word. The counts are here anyway,
         * so check, and decline the pass rather than return garbage. */
        {
            uint64_t sum = 0;
            for (size_t i = 0; i < grid; i++)
                sum += sc->h_counts[i];
            if (sum != total) {
                fprintf(stderr, "# las-gpu: %llu bucket updates in one"
                        " pass, more than it can index\n",
                        (unsigned long long) sum);
                return gpu_fib_output {};
            }
        }

        if (total > sc->updates_capacity) {
            /* Only on the first few special-q's: the high-water mark
             * drifts up, so grow generously. The counts are still
             * good; only the scatter has to be done again. */
            uint64_t want = uint64_t(total) + total / 4;
            if (want > UINT32_MAX)
                want = UINT32_MAX;
            sc->d_updates = nullptr;
            sc->updates_capacity = 0;
            if (!CUDA_ALLOC(cudaMalloc(&sc->d_updates,
                                       size_t(want) *
                                           sizeof(gpu_fib_update))))
                return gpu_fib_output {};
            sc->updates_capacity = uint32_t(want);
            fib_enqueue_scatter(sc, grid);
        }
        if (!fib_host_take(sc, total))
            return gpu_fib_output {};

        if (nhost)
            CUDA_CHECK(cudaMemcpyAsync(sc->h_host_entries, sc->d_host_entries,
                                       size_t(nhost) * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(sc->h_updates, sc->d_updates,
                                   size_t(total) * sizeof(gpu_fib_update),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        return gpu_fib_output {
            true, t->nslices, sc->plan.chunk_count,
            sc->h_counts, sc->h_offsets, sc->h_updates,
            sc->h_host_entries, nhost,
        };
    }

    void fib_release(gpu_fib_pass * pass) override
    {
        auto * sc = reinterpret_cast<cuda_fib_scratch *>(pass);
        /* A pass may be given back without having been waited for --
         * the last chunks of a special-q whose J turned out small, or
         * one prefetched for nothing. Its kernels may still be running,
         * writing to buffers that the next user of the scratch, or of
         * the walk state, is about to touch. */
        cudaSetDevice(device);
        CUDA_CHECK(cudaStreamSynchronize(sc->stream));
        sc->task = nullptr;
        sc->state = nullptr;
        fib_host_give(sc);
        std::lock_guard<std::mutex> const lock(fib_mu);
        fib_free.push_back(sc);
    }
};

/* Creating the context is the first thing that needs device memory, and
 * the one failure that is worth telling apart: other processes may give
 * the memory back. A failed attempt leaves nothing behind -- the runtime
 * tries again on the next call -- but it does forget the flags, which
 * is why they are set every time. */
bool cuda_device_usable(int device, bool * out_of_memory)
{
    auto ok = [&](cudaError_t e) {
        if (e == cudaSuccess)
            return true;
        (void) cudaGetLastError();
        if (e == cudaErrorMemoryAllocation)
            *out_of_memory = true;
        return false;
    };
    int count = 0;
    CUDA_TRY(cudaGetDeviceCount(&count));
    if (device < 0 || device >= count)
        return false;
    if (!ok(cudaSetDevice(device)))
        return false;
    /* By default a thread waiting on the device spins as long as there
     * are fewer active host threads than cores, which in las is always:
     * the pool has one thread per core, and the ones waiting on a pass
     * would burn the very cores the rest of the sieve needs. Make them
     * sleep instead. This has to be set before the context exists; if
     * some other code got there first, the default stays, which is
     * merely slower. */
    cudaError_t const e = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (e != cudaSuccess && e != cudaErrorSetOnActiveProcess)
        return false;
    (void) cudaGetLastError();
    return ok(cudaFree(nullptr)); /* forces context creation */
}

} // namespace

gpu_engine * gpu_engine_cuda_create(int device, bool * out_of_memory)
{
    bool oom = false;
    bool const usable = cuda_device_usable(device, &oom);
    if (out_of_memory)
        *out_of_memory = oom;
    if (!usable)
        return nullptr;
    return new cuda_engine(device);
}

bool gpu_engine_cuda_compiled()
{
    return true;
}
