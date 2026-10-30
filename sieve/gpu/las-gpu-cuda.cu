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

__global__ void kernel_fib_count(gpu_fib_entry const * __restrict__ entries,
                                 uint32_t const * __restrict__ entry_slice,
                                 uint32_t nentries, gpu_fib_plan plan,
                                 uint32_t nslices,
                                 uint32_t * __restrict__ counts,
                                 uint32_t * __restrict__ host_entries,
                                 unsigned long long * __restrict__ nhost)
{
    gpu_fib_fence const F = gpu_fib_fence_for(plan.logI, plan.J);
    int const logB = plan.log_bucket_region;
    uint32_t const stride = blockDim.x * gridDim.x;

    for (uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < nentries;
         k += stride) {
        gpu_fib_entry const e = entries[k];
        if (!gpu_fib_entry_is_sieved(plan, e.p))
            continue;

        gpu_ple ple {};
        int const kind = gpu_fib_prepare(&ple, e, plan.basis, F, plan.logI);
        if (kind == GPU_FIB_HOST) {
            /* sized for every entry, so this cannot overflow */
            unsigned long long const slot = atomicAdd(nhost, 1ULL);
            host_entries[slot] = k;
            continue;
        }

        uint32_t * const col = counts + entry_slice[k];
        gpu_fib_walk(ple, kind, F, [&](uint64_t x) {
            atomicAdd(&col[size_t(x >> logB) * nslices], 1u);
        });
    }
}

/* cursor[] starts out as the exclusive prefix sums of counts[]. Updates
 * past `capacity` are counted but not written; the host notices that
 * the total exceeds the capacity, grows the buffer and runs this kernel
 * again. */
__global__ void kernel_fib_scatter(gpu_fib_entry const * __restrict__ entries,
                                   uint32_t const * __restrict__ entry_slice,
                                   uint32_t nentries, gpu_fib_plan plan,
                                   uint32_t nslices,
                                   uint32_t * __restrict__ cursor,
                                   gpu_fib_update * __restrict__ updates,
                                   uint32_t capacity)
{
    gpu_fib_fence const F = gpu_fib_fence_for(plan.logI, plan.J);
    int const logB = plan.log_bucket_region;
    uint32_t const bmask = (UINT32_C(1) << logB) - 1;
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

        uint32_t * const col = cursor + entry_slice[k];
        uint16_t const hint = e.hint;
        gpu_fib_walk(ple, kind, F, [&](uint64_t x) {
            uint32_t const slot =
                atomicAdd(&col[size_t(x >> logB) * nslices], 1u);
            if (slot < capacity)
                updates[slot] = gpu_fib_update {hint, uint16_t(x & bmask)};
        });
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

struct cuda_fib_task {
    /* uploaded once, reused for every special-q: the factor base does
     * not depend on q */
    gpu_fib_entry * d_entries = nullptr;
    uint32_t * d_entry_slice = nullptr;
    uint32_t nentries = 0;
    uint32_t nslices = 0;
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

    void fib_host_take(cuda_fib_scratch * sc, uint32_t total)
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
            uint64_t want = uint64_t(total) + total / 4;
            if (want > UINT32_MAX)
                want = UINT32_MAX;
            CUDA_CHECK(cudaHostAlloc(&hb.p,
                                     size_t(want) * sizeof(gpu_fib_update),
                                     cudaHostAllocDefault));
            hb.capacity = uint32_t(want);
        }
        sc->h_updates = hb.p;
        sc->h_updates_capacity = hb.capacity;
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
        CUDA_CHECK(cudaStreamCreate(&sc->stream));
        CUDA_CHECK(cudaMalloc(&sc->d_nhost, sizeof(unsigned long long)));
        CUDA_CHECK(cudaHostAlloc(&sc->h_nhost, sizeof(unsigned long long),
                                 cudaHostAllocDefault));
        std::lock_guard<std::mutex> const lock(fib_mu);
        fib_all.push_back(sc);
        return sc;
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

        size_t const ebytes = size_t(nentries) * sizeof(gpu_fib_entry);
        CUDA_CHECK(cudaMalloc(&t->d_entries, ebytes));
        CUDA_CHECK(cudaMemcpy(t->d_entries, entries, ebytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&t->d_entry_slice,
                              size_t(nentries) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(t->d_entry_slice, es.data(),
                              size_t(nentries) * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        return reinterpret_cast<gpu_fib_task *>(t);
    }

    void fib_task_destroy(gpu_fib_task * task) override
    {
        auto * t = reinterpret_cast<cuda_fib_task *>(task);
        if (!t)
            return;
        cudaSetDevice(device);
        cudaFree(t->d_entries);
        cudaFree(t->d_entry_slice);
        delete t;
    }

    static void fib_enqueue_scatter(cuda_fib_scratch * sc, size_t grid)
    {
        cuda_fib_task const * const t = sc->task;
        CUDA_CHECK(cudaMemcpyAsync(sc->d_cursor, sc->d_offsets,
                                   grid * sizeof(uint32_t),
                                   cudaMemcpyDeviceToDevice, sc->stream));
        kernel_fib_scatter<<<sc->grid_e, 128, 0, sc->stream>>>(
            t->d_entries, t->d_entry_slice, t->nentries, sc->plan,
            t->nslices, sc->d_cursor, sc->d_updates, sc->updates_capacity);
        CUDA_CHECK(cudaGetLastError());
    }

    /* Everything up to and including the scatter goes into the stream
     * here, with no synchronisation: the device needs nothing from the
     * host once it has the plan. Only copying the updates back has to
     * wait for fib_wait(), because their number is not known until the
     * count is done. */
    gpu_fib_pass * fib_launch(gpu_fib_task * task,
                              gpu_fib_plan const & plan) override
    {
        auto * t = reinterpret_cast<cuda_fib_task *>(task);
        CUDA_CHECK(cudaSetDevice(device));
        cuda_fib_scratch * const sc = fib_scratch_take();
        cudaStream_t const stream = sc->stream;
        sc->task = t;
        sc->plan = plan;

        size_t const grid = size_t(t->nslices) * plan.n_buckets;

        if (grid + 1 > sc->grid_capacity) {
            cudaFree(sc->d_counts);
            cudaFree(sc->d_offsets);
            cudaFree(sc->d_cursor);
            cudaFree(sc->d_scan_temp);
            cudaFreeHost(sc->h_counts);
            cudaFreeHost(sc->h_offsets);
            size_t const n = grid + 1;
            CUDA_CHECK(cudaMalloc(&sc->d_counts, n * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&sc->d_offsets, n * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&sc->d_cursor, n * sizeof(uint32_t)));
            CUDA_CHECK(cudaHostAlloc(&sc->h_counts, n * sizeof(uint32_t),
                                     cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc(&sc->h_offsets, n * sizeof(uint32_t),
                                     cudaHostAllocDefault));
            sc->scan_temp_bytes = 0;
            CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                nullptr, sc->scan_temp_bytes, sc->d_counts, sc->d_offsets,
                n));
            CUDA_CHECK(cudaMalloc(&sc->d_scan_temp, sc->scan_temp_bytes));
            sc->grid_capacity = n;
        }
        if (t->nentries > sc->host_capacity) {
            cudaFree(sc->d_host_entries);
            cudaFreeHost(sc->h_host_entries);
            CUDA_CHECK(cudaMalloc(&sc->d_host_entries,
                                  size_t(t->nentries) * sizeof(uint32_t)));
            CUDA_CHECK(cudaHostAlloc(&sc->h_host_entries,
                                     size_t(t->nentries) * sizeof(uint32_t),
                                     cudaHostAllocDefault));
            sc->host_capacity = t->nentries;
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

        kernel_fib_count<<<sc->grid_e, 128, 0, stream>>>(
            t->d_entries, t->d_entry_slice, t->nentries, plan, t->nslices,
            sc->d_counts, sc->d_host_entries, sc->d_nhost);
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
        size_t const grid = size_t(t->nslices) * sc->plan.n_buckets;

        CUDA_CHECK(cudaStreamSynchronize(stream));
        uint32_t const total = sc->h_offsets[grid];
        uint64_t const nhost = *sc->h_nhost;

        if (total > sc->updates_capacity) {
            /* Only on the first few special-q's: the high-water mark
             * drifts up, so grow generously. The counts are still
             * good; only the scatter has to be done again. */
            uint64_t want = uint64_t(total) + total / 4;
            if (want > UINT32_MAX)
                want = UINT32_MAX;
            cudaFree(sc->d_updates);
            CUDA_CHECK(cudaMalloc(&sc->d_updates,
                                  size_t(want) * sizeof(gpu_fib_update)));
            sc->updates_capacity = uint32_t(want);
            fib_enqueue_scatter(sc, grid);
        }
        fib_host_take(sc, total);

        if (nhost)
            CUDA_CHECK(cudaMemcpyAsync(sc->h_host_entries, sc->d_host_entries,
                                       size_t(nhost) * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(sc->h_updates, sc->d_updates,
                                   size_t(total) * sizeof(gpu_fib_update),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        return gpu_fib_output {
            t->nslices, sc->plan.n_buckets,
            sc->h_counts, sc->h_offsets, sc->h_updates,
            sc->h_host_entries, nhost,
        };
    }

    void fib_release(gpu_fib_pass * pass) override
    {
        auto * sc = reinterpret_cast<cuda_fib_scratch *>(pass);
        sc->task = nullptr;
        fib_host_give(sc);
        std::lock_guard<std::mutex> const lock(fib_mu);
        fib_free.push_back(sc);
    }
};

bool cuda_device_usable(int device)
{
    int count = 0;
    CUDA_TRY(cudaGetDeviceCount(&count));
    if (device < 0 || device >= count)
        return false;
    CUDA_TRY(cudaSetDevice(device));
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
    CUDA_TRY(cudaFree(nullptr)); /* forces context creation */
    return true;
}

} // namespace

gpu_engine * gpu_engine_cuda_create(int device)
{
    if (!cuda_device_usable(device))
        return nullptr;
    return new cuda_engine(device);
}

bool gpu_engine_cuda_compiled()
{
    return true;
}
