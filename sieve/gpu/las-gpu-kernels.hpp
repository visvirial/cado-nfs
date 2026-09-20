#ifndef CADO_LAS_GPU_KERNELS_HPP
#define CADO_LAS_GPU_KERNELS_HPP

#include <cmath>
#include <cstdint>

#include "las-gpu-arch.hpp"
#include "las-gpu-plan.hpp"

/* The arithmetic of the two kernels, written once and compiled twice:
 * once as ordinary host code (las-gpu-cpu.cpp) and once as device code
 * (las-gpu-cuda.cu). Keeping a single definition is what makes the CPU
 * backend a meaningful oracle for the CUDA one.
 *
 * Everything here is a pure function of one output cell. That is the
 * whole difference with the CPU sieve code it replaces: lognorm_smart
 * walks a line and emits runs of equal values, which is excellent for a
 * single core and useless for 10^4 threads.
 */

/* }}} */
/* {{{ the log2 approximation, as in las-norms.cpp */

/* Extract the top 32 bits of the IEEE-754 representation of z and
 * return them as a double. For z > 0 this is an affine function of
 * log2(z), up to an error < 0.0861 -- see the long comment at the top of
 * las-norms.cpp.
 */
CADO_GPU_FN double gpu_lg2_raw(double z)
{
#if defined(__CUDA_ARCH__)
    return (double) (uint32_t) ((uint64_t) __double_as_longlong(z) >> 32);
#else
    uint64_t bits;
    __builtin_memcpy(&bits, &z, sizeof(bits));
    return (double) (uint32_t) (bits >> 32);
#endif
}

/* Same as COMPUTE_Y() in lognorm_smart::fill_rat_inner. The +1. keeps the
 * argument away from zero, which matters because F(i,j) really does
 * vanish on the sieve area.
 */
CADO_GPU_FN unsigned char gpu_compute_y(double g, double offset,
                                        double modscale)
{
    return (unsigned char) ((gpu_lg2_raw(g + 1.) - offset) * modscale);
}

/* }}} */
/* {{{ region geometry -- mirrors LOGNORM_FILL_COMMON_DEFS */

CADO_GPU_FN gpu_region_geometry gpu_region_geometry_for(int logI,
                                                        int log_bucket_region,
                                                        unsigned int N)
{
    int const llpr = log_bucket_region > logI ? log_bucket_region - logI : 0;
    int const lrpl = logI > log_bucket_region ? logI - log_bucket_region : 0;
    unsigned int const regions_per_line = 1U << lrpl;
    unsigned int const rank = N & (regions_per_line - 1);

    gpu_region_geometry g;
    g.j0 = (N >> lrpl) << llpr;
    g.j1 = g.j0 + (1U << llpr);
    g.I = 1 << logI;
    g.i0 = (int) (rank << log_bucket_region) - g.I / 2;
    g.i1 = g.i0 + (1 << (log_bucket_region < logI ? log_bucket_region : logI));
    g.has_origin = (g.j0 == 0) && (rank == (regions_per_line - 1) / 2);
    return g;
}

/* }}} */
/* {{{ norm of a single cell */

/* Locate the affine piece of G that lognorm_smart::fill_alg would be
 * using at abscissa i on line j.
 *
 * fill_alg walks the pieces in order, keeping Gi0 = max(i0, previous
 * Gi1); piece k covers [Gi0, Gi1_k) with
 *     Gi1_k = (int) (j * r1_k + (j * r1_k >= 0)).
 * Since Gi1_k is non-decreasing in k, "the piece covering i" is simply
 * the first k with Gi1_k > i, which is what we binary-search for.
 */
CADO_GPU_FN int gpu_norm_segment_index(gpu_norm_segment const * segments,
                                       int nsegments, double jj, int i)
{
    int lo = 0, hi = nsegments - 1;
    while (lo < hi) {
        int const mid = lo + (hi - lo) / 2;
        double const jr1 = jj * segments[mid].r1;
        int const Gi1 = (int) (jr1 + (jr1 >= 0));
        if (Gi1 > i)
            hi = mid;
        else
            lo = mid + 1;
    }
    /* If i lies beyond the last endpoint we clamp to the last piece.
     * lognorm_smart leaves such cells untouched (G is built to cover
     * [-I,I], so this is not expected to happen); extrapolating the last
     * piece is the least surprising thing to do. */
    return lo;
}

/* Value of the lognorm at (i, j). j must be >= 1. */
CADO_GPU_FN unsigned char gpu_norm_at(gpu_norm_plan const & P, int i,
                                      unsigned int j)
{
    double u0, u1;

    if (P.kind == gpu_norm_plan::KIND_RAT) {
        u0 = P.u0;
        u1 = P.u1;
    } else if (P.kind == gpu_norm_plan::KIND_SIQS) {
        /* J == 1: every line uses the same approximation, evaluated at
         * j = 1, and lognorm_smart::fill_siqs copies line 0 over. */
        int const k = gpu_norm_segment_index(P.segments, P.nsegments, 1., i);
        u0 = P.segments[k].u;
        u1 = P.segments[k].v;
        j = 1;
    } else {
        /* F(i,j) = j^d * F(i/j, 1), and with g = u + v*x we get
         *     g(i/j) * j^d = (u * j^(d-1)) * j + (v * j^(d-1)) * i.
         * The repeated multiplication below reproduces the exact
         * rounding of the mj1 loop in lognorm_smart::fill_alg. */
        double mj1 = j;
        for (int d = P.degree; --d > 1;)
            mj1 *= j;
        int const k =
            gpu_norm_segment_index(P.segments, P.nsegments, (double) j, i);
        u0 = P.segments[k].u * mj1;
        u1 = P.segments[k].v * mj1;
    }

    double const g = fabs(u0 * (double) j + u1 * (double) i);
    return gpu_compute_y(g, P.offset, P.modscale);
}

/* Value of cell x of bucket region number N, 0 <= x < 2^log_bucket_region.
 * This is the exact contract of lognorm_base::fill().
 */
CADO_GPU_FN unsigned char gpu_norm_cell(gpu_norm_plan const & P,
                                        gpu_region_geometry const & geo,
                                        uint32_t x)
{
    int const line_len = geo.i1 - geo.i0;
    uint32_t const line = x / (uint32_t) line_len;
    uint32_t const in_line = x % (uint32_t) line_len;
    unsigned int const j = geo.j0 + line;
    int const i = geo.i0 + (int) in_line;

    /* The line j = 0 carries no useful norm: (i,0) is a valid pair only
     * for i = +-1, and we keep the positive one. */
    if (P.kind != gpu_norm_plan::KIND_SIQS && geo.has_origin && j == 0)
        return (i == 1) ? P.origin_value : 255;

    return gpu_norm_at(P, i, j);
}

/* }}} */
/* {{{ survivor search */

CADO_GPU_FN uint32_t gpu_gcd(uint32_t a, uint32_t b)
{
    if (a == 0) return b;
    if (b == 0) return a;
    while (b) {
        uint32_t const t = a % b;
        a = b;
        b = t;
    }
    return a;
}

/* Result of examining one cell. The CPU code has the side effect of
 * writing 255 over every non-survivor of the Sf side, and callers
 * downstream (purge_buckets, resieve, cofactoring) rely on it.
 */
struct gpu_survivor_verdict {
    bool survives;
};

/* Reproduces, for the single cell at offset in_line of line j, the
 * combined effect of unsieve_not_coprime_line() and
 * search_survivors_in_line1() from las-unsieve.cpp.
 *
 * The two of them together discard (i,j) whenever some odd prime divides
 * both i and j -- but not quite: for a prime p > 7 that is also above
 * the trial-division threshold, unsieve_one_prime() only walks the odd
 * offsets when j is even, so cells with i and j both even survive that
 * particular test. We reproduce that asymmetry rather than fix it,
 * because the point here is to be a drop-in replacement.
 */
CADO_GPU_FN bool gpu_survivor_coprimality(gpu_survivor_region const & R,
                                          gpu_survivor_tables const & T,
                                          uint32_t in_line, unsigned int j,
                                          bool & unsieve_marked)
{
    int const i_signed = R.geo.i0 + (int) in_line;
    uint32_t const i = (uint32_t) (i_signed < 0 ? -i_signed : i_signed);

    bool const j_even = (j & 1) == 0;
    bool const x_odd = (in_line & 1) != 0;

    unsieve_marked = false;
    bool divides_small = false;

    /* Walk the distinct odd prime factors of j, largest first. */
    uint32_t c = j;
    while ((c & 1) == 0)
        c >>= 1;

    while (c > 1) {
        gpu_jdiv_entry const e = T.jdiv[c];
        uint32_t const p = e.p;
        if (p < 3)
            break;
        c = e.cof;

        /* p | i, by the usual multiply-by-the-inverse test. */
        bool const div = (i * e.inv) <= e.bound;
        if (!div)
            continue;

        if (p > R.td_max) {
            if (p > 7) {
                /* unsieve_one_prime() */
                if (!j_even || x_odd)
                    unsieve_marked = true;
            } else {
                /* unsieve_3 / unsieve_5 / unsieve_7 */
                unsieve_marked = true;
            }
        } else {
            /* the switch() in search_survivors_in_line1() */
            divides_small = true;
        }
    }

    return divides_small;
}

/* Full per-cell predicate. s0/s1 are the norms as they stand on entry;
 * *sf_out is set to 255 when the cell must be marked as dead.
 */
CADO_GPU_FN bool gpu_survivor_cell(gpu_survivor_region const & R,
                                   gpu_survivor_tables const & T,
                                   unsigned char s0, unsigned char s1,
                                   uint32_t x, unsigned char & sf_out)
{
    int const line_len = R.geo.i1 - R.geo.i0;
    uint32_t const in_line = x % (uint32_t) line_len;
    unsigned int const j = R.geo.j0 + x / (uint32_t) line_len;

    unsigned char const b0 = R.bound[0];
    unsigned char const b1 = R.bound[1];
    unsigned char const sf = (R.side_of_Sf == 0) ? s0 : s1;
    unsigned char const bf = R.bound[R.side_of_Sf];

    sf_out = 255;

    /* Line j = 0: only (i,j) = (-1,0) can ever give a relation, and
     * las-unsieve.cpp keeps it at offset 1 - i0. */
    if (j == 0 && R.sublat_m == 0) {
        if (!(R.geo.i0 <= 0 && R.geo.i1 > 0))
            return false;
        if (in_line != (uint32_t) (1 - R.geo.i0))
            return false;
        bool const ok = (R.nsides == 2) ? (s0 <= b0 && s1 <= b1) : (sf <= bf);
        if (!ok)
            return false;
        sf_out = sf;
        return true;
    }

    if (R.sublat_m != 0) {
        bool const ok = (R.nsides == 2) ? (s0 <= b0 && s1 <= b1) : (sf <= bf);
        if (!ok)
            return false;
        int const t = (int) R.sublat_m * (R.geo.i0 + (int) in_line);
        /* Note the two branches of las-unsieve.cpp do not bracket this
         * the same way; we follow each of them. */
        uint32_t i;
        if (R.nsides == 2) {
            int const v = t + (int) R.sublat_i0;
            i = (uint32_t) (v < 0 ? -v : v);
        } else {
            i = (uint32_t) (t < 0 ? -t : t) + R.sublat_i0;
        }
        uint32_t const jj = R.sublat_m * j + R.sublat_j0;
        if (((jj % 2) == 0 && (i % 2) == 0) || gpu_gcd(i, jj) != 1)
            return false;
        sf_out = sf;
        return true;
    }

    /* The bound pattern that search_survivors_in_line*_sse2() compares
     * against has its even bytes zeroed when j is even (even_masks[] in
     * las-unsieve-sse2.cpp): for such j no cell at an even offset can be
     * coprime, so the whole even half of the line dies whatever the
     * norms say. Note that the reference keys this on the offset within
     * the line, not on the parity of i -- the two only differ for the
     * odd i0 that a logI of 1 would give, and we follow the reference.
     */
    bool const killed_by_parity = ((j & 1) == 0) && ((in_line & 1) == 0);

    bool unsieve_marked;
    bool const divides_small =
        gpu_survivor_coprimality(R, T, in_line, j, unsieve_marked);

    /* unsieve_not_coprime_line() runs first and writes 255 into Sf; the
     * bound test below then sees that value. */
    unsigned char const sf_eff = unsieve_marked ? 255 : sf;
    unsigned char const s0_eff =
        (R.side_of_Sf == 0 && unsieve_marked) ? 255 : s0;
    unsigned char const s1_eff =
        (R.side_of_Sf == 1 && unsieve_marked) ? 255 : s1;

    bool const ok =
        (R.nsides == 2) ? (s0_eff <= b0 && s1_eff <= b1) : (sf_eff <= bf);
    if (!ok || divides_small || killed_by_parity)
        return false;

    sf_out = sf_eff;
    return true;
}

/* }}} */

#endif /* CADO_LAS_GPU_KERNELS_HPP */
