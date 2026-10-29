#ifndef CADO_LAS_GPU_FIB_PLAN_HPP
#define CADO_LAS_GPU_FIB_PLAN_HPP

#include <cstdint>

/* Plain-old-data description of one bucket-sieving pass, shared verbatim
 * between the host and the device. Same rules as las-gpu-plan.hpp: these
 * must stay trivially copyable and free of anything nvcc dislikes.
 */

/* The q-lattice basis, which is the only thing that changes from one
 * special-q to the next. Mirrors the four coordinates of
 * qlattice_basis; the 31-bit root transform needs nothing else.
 */
struct gpu_qlattice {
    int64_t a0, b0, a1, b1;
};

/* One (factor base entry, root) pair, flattened. The factor base itself
 * does not depend on the special-q, so the whole array is uploaded once
 * and then reused; only gpu_qlattice changes per special-q.
 *
 * hint is the index of the entry within its slice, which is what ends up
 * in the bucket update and what resieving uses to recover p.
 */
struct gpu_fib_entry {
    uint32_t p;
    uint32_t invq;      /* -1/p mod 2^32 */
    uint32_t r;         /* root in the (a,b) plane, affine */
    uint16_t hint;      /* slice_offset_t */
    uint16_t pad;
};

/* A contiguous run of gpu_fib_entry that shares one slice index, hence
 * one logp. Bucket updates must stay grouped by slice, because that is
 * how apply_one_bucket recovers logp.
 */
struct gpu_fib_slice {
    uint32_t first;     /* index into the entry array */
    uint32_t count;
    uint32_t slice_index;
    uint32_t pad;
};

/* Everything that depends on the special-q but not on the slice. */
struct gpu_fib_plan {
    int logI;
    int log_bucket_region;
    uint32_t J;
    uint32_t n_buckets;

    gpu_qlattice basis;

    /* The prime factors of the special-q, which the ordinary code skips
     * via Q.is_coprime_to(). There is almost always exactly one.
     */
    uint32_t q_factors[8];
    int n_q_factors;
};

/* Mirror of bucket_update_t<1, shorthint_t>. The field order is the one
 * the real type gets from inheriting shorthint_t before
 * bare_bucket_update_t; las-fill-in-buckets-gpu.cpp static_asserts it.
 */
struct gpu_fib_update {
    uint16_t hint;
    uint16_t x;
};

#endif /* CADO_LAS_GPU_FIB_PLAN_HPP */
