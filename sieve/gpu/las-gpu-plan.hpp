#ifndef CADO_LAS_GPU_PLAN_HPP
#define CADO_LAS_GPU_PLAN_HPP

#include <cstdint>

/* Plain-old-data descriptions of the work that the norm-initialisation
 * and survivor-search kernels have to do. These structures are shared
 * verbatim between the host and the device, so they must stay trivially
 * copyable and must not contain anything that nvcc cannot chew on.
 */

/* Mirror of las-config.hpp's LOGNORM_GUARD_BITS. We do not include that
 * header here (see las-gpu-arch.hpp for why); las-norms-gpu.cpp has a
 * static_assert that keeps the two in sync.
 */
#define CADO_GPU_LOGNORM_GUARD_BITS 1

/* One affine piece of the piecewise-linear approximation G of F(x,1):
 * on [r0, r1) we have F(x,1) ~= u + v*x.
 */
struct gpu_norm_segment {
    double r0, r1;
    double u, v;
};

/* Everything the norm kernel needs that does not depend on the bucket
 * region number N.
 */
struct gpu_norm_plan {
    int logI;
    int log_bucket_region;

    /* Which of the three code paths of lognorm_smart::fill applies.
     * Mirrors the J == 1 / degree > 1 / else dispatch.
     */
    enum kind_t : int { KIND_RAT = 0, KIND_ALG = 1, KIND_SIQS = 2 };
    int kind;

    int degree;          /* fijd.degree() */
    double scale;
    double modscale;     /* scale / 0x100000 */
    double offset;       /* 0x3FF00000 - LOGNORM_GUARD_BITS / modscale */

    /* Value written at (i,j) = (1,0) when the region contains the
     * origin; precomputed on the host because it needs log2().
     */
    unsigned char origin_value;

    /* KIND_RAT only: F(i,j) = u0*j + u1*i */
    double u0, u1;

    /* KIND_ALG / KIND_SIQS: see gpu_norm_segment. Owned by the engine,
     * points into device memory for the CUDA backend.
     */
    gpu_norm_segment const * segments;
    int nsegments;
};

/* Geometry of one bucket region, derived from N exactly the way the
 * LOGNORM_FILL_COMMON_DEFS macro does it in las-norms.cpp.
 */
struct gpu_region_geometry {
    int i0, i1;
    unsigned int j0, j1;
    int I;
    int has_origin;
};

/* Mirror of j_divisibility_helper::entry */
struct gpu_jdiv_entry {
    uint32_t p, cof, inv, bound;
};

/* Mirror of unsieve_data::entry */
struct gpu_unsieve_entry {
    uint32_t lpf, cof;
};

/* Tables needed by the survivor kernel; they only depend on J, so they
 * are uploaded once per special-q (in fact once per las_info).
 */
struct gpu_survivor_tables {
    gpu_jdiv_entry const * jdiv;
    uint32_t jdiv_size;
    gpu_unsieve_entry const * unsieve;
    uint32_t unsieve_Jmax;
};

/* Everything else the survivor kernel needs. */
struct gpu_survivor_region {
    int nsides;                 /* 1 or 2 */
    int side_of_Sf;             /* which of S0/S1 gets the 255 markers */
    unsigned char bound[2];
    gpu_region_geometry geo;
    int logI;
    uint32_t td_max;            /* conf.unsieve_thresh */
    uint32_t sublat_m, sublat_i0, sublat_j0;
};

#endif /* CADO_LAS_GPU_PLAN_HPP */
