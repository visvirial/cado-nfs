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

#include <cuda_runtime.h>
#include <cub/cub.cuh>
/* cub::CountingInputIterator is gone in cccl 3 (cuda 13); the thrust
 * spelling is the one that works with both. */
#include <thrust/iterator/counting_iterator.h>

#include "las-gpu-engine.hpp"
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
/* {{{ tasks */

struct cuda_norm_task {
    gpu_norm_plan plan;          /* .segments points to device memory */
    gpu_norm_segment * d_segments = nullptr;
    unsigned char * d_S = nullptr;
    size_t d_S_bytes = 0;
};

/* How many survivors we fetch without being asked twice. A 64kB region
 * yields a few hundred at the densities las actually sieves at, so this
 * covers it with room to spare; the slow path below catches the rest.
 */
#define CUDA_SURVIVOR_PREFETCH 4096

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
        cudaFree(t->d_S);
        delete t;
    }

    void norm_fill(gpu_norm_task * task, unsigned char * S, unsigned int N0,
                   unsigned int nregions) override
    {
        auto * t = reinterpret_cast<cuda_norm_task *>(task);
        CUDA_CHECK(cudaSetDevice(device));

        size_t const bytes = (size_t(nregions) << t->plan.log_bucket_region);
        if (bytes > t->d_S_bytes) {
            cudaFree(t->d_S);
            CUDA_CHECK(cudaMalloc(&t->d_S, bytes));
            t->d_S_bytes = bytes;
        }

        int const block = 256;
        int grid = int((bytes + block - 1) / block);
        if (grid > 65535)
            grid = 65535;

        kernel_fill_norms<<<grid, block>>>(t->d_S, t->plan, N0, nregions);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(S, t->d_S, bytes, cudaMemcpyDeviceToHost));
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
};

bool cuda_device_usable(int device)
{
    int count = 0;
    CUDA_TRY(cudaGetDeviceCount(&count));
    if (device < 0 || device >= count)
        return false;
    CUDA_TRY(cudaSetDevice(device));
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
