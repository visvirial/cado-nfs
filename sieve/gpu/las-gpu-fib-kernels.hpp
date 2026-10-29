#ifndef CADO_LAS_GPU_FIB_KERNELS_HPP
#define CADO_LAS_GPU_FIB_KERNELS_HPP

#include <cstdint>

#include "las-gpu-arch.hpp"
#include "las-gpu-fib-plan.hpp"

/* Device-compatible transcription of the arithmetic that
 * fill_in_buckets_toplevel() needs, from the factor base entry all the
 * way to the stream of bucket updates.
 *
 * Every function here has a counterpart in the ordinary sieve code, and
 * the two must agree exactly -- not approximately, as for the lognorms,
 * but bit for bit, because a single missing update changes the set of
 * relations. The counterparts are, in order:
 *
 *      gpu_redc_u32, gpu_redc_32,
 *      gpu_varredc_u32, gpu_addmod_u32,
 *      gpu_invmod_redc_32              sieve/las-arith.hpp
 *      gpu_fb_root_in_qlattice         sieve/las-fbroot-qlattice.hpp
 *      gpu_plattice::reduce            sieve/las-reduce-plattice-simplistic.hpp
 *      gpu_plattice_enumerator         sieve/las-plattice.hpp
 *
 * The host versions of the first group have hand-written assembly
 * variants; tests/sieve/test-reduce-plattice and torture-redc already
 * pin those to the portable C, which is what we transcribe.
 *
 * As everywhere else in this directory, no cado-nfs header may be
 * included: nvcc compiles this file.
 */

/* {{{ REDC arithmetic */

/* redc_no_carry(p): p < 2^31 lets the caller skip the carry handling.
 * We keep the check as a run-time argument rather than a template
 * parameter, because a kernel sees a whole slice at once and we would
 * rather not specialise it twice.
 */
CADO_GPU_FN bool gpu_redc_no_carry(uint32_t p)
{
    return p < (UINT32_C(1) << 31);
}

/* x/2^32 mod p, for 0 <= x < 2^32*p. Mirrors redc_u32(). */
CADO_GPU_FN uint32_t gpu_redc_u32(uint64_t x, uint32_t p, uint32_t invp,
                                  bool carrycheck)
{
    uint32_t t = uint32_t(x) * invp;
    uint64_t const tp = uint64_t(t) * uint64_t(p);
    uint64_t xtp = x;

    t = 0;
    xtp += tp;
    if (carrycheck && xtp < tp)
        t = p;

    uint32_t const u = uint32_t(xtp >> 32);
    return u - ((u >= p) ? p : t);
}

/* x/2^32 mod p, for x in ]-2^32*p, 2^32*p[. Mirrors redc_32(). */
CADO_GPU_FN uint32_t gpu_redc_32(int64_t x, uint32_t p, uint32_t invp,
                                 bool carrycheck)
{
    uint32_t t = uint32_t(uint64_t(x)) * invp;
    uint64_t const tp = uint64_t(t) * uint64_t(p);
    uint64_t xtp = (x >= 0) ? uint64_t(x) : uint64_t(x) + (uint64_t(p) << 32);

    t = 0;
    xtp += tp;
    if (carrycheck && xtp < tp)
        t = p;

    uint32_t const u = uint32_t(xtp >> 32);
    return u - ((u >= p) ? p : t);
}

/* x/2^s mod p, for x < p and s < 32. Mirrors varredc_u32(). */
CADO_GPU_FN uint32_t gpu_varredc_u32(uint32_t x, uint32_t p, uint32_t invp,
                                     unsigned s)
{
    uint32_t const mask = (UINT32_C(1) << s) - 1;
    uint32_t const t = (x * invp) & mask;
    uint64_t const xtp = uint64_t(x) + uint64_t(t) * uint64_t(p);
    return uint32_t(xtp >> s);
}

/* a+b mod m, for a, b < m. Mirrors addmod_u32(). */
CADO_GPU_FN uint32_t gpu_addmod_u32(uint32_t a, uint32_t b, uint32_t m)
{
    return (b >= m - a) ? (b - (m - a)) : (a + b);
}

CADO_GPU_FN unsigned gpu_ctz32(uint32_t x)
{
#if defined(__CUDA_ARCH__)
    return unsigned(__ffs(int(x))) - 1u;
#elif defined(__GNUC__) || defined(__clang__)
    return unsigned(__builtin_ctz(x));
#else
    unsigned n = 0;
    while (!(x & 1u)) { x >>= 1; n++; }
    return n;
#endif
}

/* 2^32/a mod b, or 0 when a is not invertible. b must be odd, a < b.
 * Mirrors invmod_redc_32(); the binary-gcd loop below is the portable
 * branch of it, the cmov assembly being an optimisation only.
 */
CADO_GPU_FN uint32_t gpu_invmod_redc_32(uint32_t a, uint32_t orig_b,
                                        uint32_t invb)
{
    uint32_t b = orig_b;
    if (!a)
        return a;

    uint32_t const p = b;
    uint32_t u = 1, v = 0;
    unsigned lsh = gpu_ctz32(a);
    unsigned t = lsh;
    a >>= lsh;

    for (;;) {
        uint32_t const diff1 = b - a;
        if (diff1 == 0)
            break;
        lsh = gpu_ctz32(diff1);
        uint32_t const d1 = diff1 >> lsh;
        uint32_t const v2 = v << lsh;
        uint32_t const d2 = (a - b) >> lsh;
        v += u;
        u <<= lsh;
        t += lsh;
        if (a < b) {
            b = d1;
        } else {
            a = d2;
            u = v;
            v = v2;
        }
    }

    if (a != 1)
        return 0;

    /* the inverse is u/2^t; we want u/2^32 */
    if (t >= 32) {
        u = gpu_varredc_u32(u, orig_b, invb, t - 32);
    } else {
        while (t++ < 32)
            u = gpu_addmod_u32(u, u, p);
    }
    return u;
}

/* }}} */
/* {{{ root transform */

/* (R*b1-a1)/(a0-R*b0) mod p, for an affine root R given in the (a,b)
 * plane. Mirrors fb_root_in_qlattice_31bits<>() restricted to the
 * affine-input case, which is all that simple factor base entries ever
 * have.
 *
 * Returns 0 for an affine transformed root, 1 for a projective one, and
 * -1 when neither numerator nor denominator is invertible. The last case
 * means p divides the special-q; the host filters those entries out with
 * is_coprime_to() before we ever see them, exactly as the ordinary code
 * path does, but we report it rather than return nonsense so that the
 * caller can hand the entry back to the CPU and get its diagnostic.
 */
CADO_GPU_FN int gpu_fb_root_in_qlattice(uint32_t * out, uint32_t p,
                                        uint32_t R, uint32_t invp,
                                        gpu_qlattice const & basis)
{
    bool const cc = !gpu_redc_no_carry(p);

    int64_t const aux1 = int64_t(R) * basis.b1 - basis.a1;
    int64_t const aux2 = basis.a0 - int64_t(R) * basis.b0;

    uint32_t const u = gpu_redc_32(aux1, p, invp, cc);
    uint32_t const v = gpu_redc_32(aux2, p, invp, cc);

    uint32_t inv = gpu_invmod_redc_32(v, p, invp);
    if (inv) {
        *out = gpu_redc_u32(uint64_t(inv) * uint64_t(u), p, invp, cc);
        return 0;
    }
    /* the root is projective in the (i,j) plane */
    inv = gpu_invmod_redc_32(u, p, invp);
    if (!inv)
        return -1;
    *out = gpu_redc_u32(uint64_t(inv) * uint64_t(v), p, invp, cc);
    return 1;
}

/* }}} */
/* {{{ p-lattice reduction */

/* Mirror of plattice_info. Same field names, same meaning: the basis is
 * <(-mi0,j0), (i1,j1)>.
 */
struct gpu_plattice {
    uint32_t mi0, j0, i1, j1;

    CADO_GPU_FN int32_t get_i0() const { return -int32_t(mi0); }
    CADO_GPU_FN int32_t get_i1() const { return int32_t(i1); }

    CADO_GPU_FN bool is_discarded() const { return mi0 == 0 && j0 == 0; }

    CADO_GPU_FN bool needs_special_treatment(uint32_t I) const
    {
        return (i1 == 0 || (j1 > 1 && mi0 < I));
    }

    CADO_GPU_FN void reduce_with_vertical_vector(uint32_t I)
    {
        uint32_t const a = (I - 1) % mi0;
        uint32_t mi2 = (I - 1) - a - i1;
        uint32_t const j2 = j1;
        if (mi0 + mi2 < I)
            mi2 += mi0;
        i1 = mi0;
        j1 = j0;
        mi0 = mi2;
        j0 = j2;
    }

    /* Mirror of reduce_plattice_simplistic(). The production host build
     * uses the assembly variant instead; tests/sieve/test-reduce-plattice
     * is what guarantees the two agree. */
    CADO_GPU_FN void reduce(uint32_t I)
    {
        if (needs_special_treatment(I)) {
            reduce_with_vertical_vector(I);
            return;
        }

        for (;;) {
            if (i1 < I) {
                if (i1 == 0) {
                    j0 = j1 - j0;
                    reduce_with_vertical_vector(I);
                    return;
                }
                uint32_t const a = (mi0 + i1 - I) / i1;
                mi0 -= a * i1;
                j0 += a * j1;
                return;
            }
            {
                uint32_t const k = mi0 / i1;
                mi0 -= k * i1;
                j0 += k * j1;
            }
            if (mi0 < I) {
                if (mi0 == 0) {
                    mi0 = i1;
                    i1 = j0; j0 = j1; j1 = i1;
                    i1 = 0;
                    reduce_with_vertical_vector(I);
                    return;
                }
                uint32_t const a = (mi0 + i1 - I) / mi0;
                i1 -= a * mi0;
                j1 += a * j0;
                return;
            }
            {
                uint32_t const k = i1 / mi0;
                i1 -= k * mi0;
                j1 += k * j0;
            }
        }
    }

    /* The initial, unreduced basis, for an affine or projective root.
     * Mirrors plattice_info::initial_basis(). */
    CADO_GPU_FN void initial_basis(uint32_t q, uint32_t r, bool proj)
    {
        if (!proj) {
            mi0 = q;
            j0 = 0;
            i1 = r;
            j1 = 1;
        } else {
            /* xgcd: g = gcd(r,q), t with t*r == g mod q */
            uint32_t a = r, b = q;
            int64_t u0 = 1, u1 = 0;
            while (b) {
                uint32_t const k = a / b;
                uint32_t const t = a - k * b;
                a = b; b = t;
                int64_t const w = u0 - int64_t(k) * u1;
                u0 = u1; u1 = w;
            }
            uint32_t const g = a;
            int64_t t = u0 % int64_t(q);
            if (t < 0)
                t += q;
            mi0 = q / g;
            j0 = 0;
            i1 = uint32_t(t);
            j1 = g;
        }
    }

    CADO_GPU_FN uint32_t get_bound_step(int) const { return mi0; }

    CADO_GPU_FN uint32_t get_bound_warp(int logI) const
    {
        if (i1 >> logI)
            return 0;
        return (1U << logI) - uint32_t(get_i1());
    }

    CADO_GPU_FN uint64_t get_inc_warp(int logI) const
    {
        return (uint64_t(j0) << logI) + uint64_t(int64_t(get_i0()));
    }

    CADO_GPU_FN uint64_t get_inc_step(int logI) const
    {
        if (i1 >> logI)
            return UINT64_MAX;
        return (uint64_t(j1) << logI) + uint64_t(int64_t(get_i1()));
    }

    CADO_GPU_FN bool is_vertical_line(int logI) const { return i1 >> logI; }
};

/* }}} */
/* {{{ enumeration */

/* Mirror of plattice_enumerator::fence */
struct gpu_fib_fence {
    uint32_t maskI;
    uint64_t even_mask;
    uint64_t end;
};

CADO_GPU_FN gpu_fib_fence gpu_fib_fence_for(int logI, uint32_t J)
{
    gpu_fib_fence F;
    F.maskI = (1U << logI) - 1U;
    F.even_mask = (uint64_t(1) << logI) | uint64_t(1);
    F.end = uint64_t(J) << logI;
    return F;
}

/* Mirror of plattice_enumerator, minus the hint, which the caller keeps
 * on the side.
 */
struct gpu_ple {
    uint64_t inc_warp, inc_step;
    uint32_t bound_step, bound_warp;
    uint64_t x;

    CADO_GPU_FN void init(gpu_plattice const & pli, int logI)
    {
        inc_warp = pli.get_inc_warp(logI);
        inc_step = pli.get_inc_step(logI);
        bound_step = pli.get_bound_step(logI);
        bound_warp = pli.get_bound_warp(logI);
        x = uint64_t(1) << (logI - 1);
    }

    CADO_GPU_FN void next(gpu_fib_fence const & F)
    {
        uint32_t const i = uint32_t(x & F.maskI);
        if (i >= bound_warp)
            x += inc_warp;
        if (i < bound_step)
            x += inc_step;
    }

    CADO_GPU_FN bool done(gpu_fib_fence const & F) const { return x >= F.end; }

    CADO_GPU_FN bool probably_coprime(gpu_fib_fence const & F) const
    {
        return (x & F.even_mask) != 0;
    }

    CADO_GPU_FN bool is_projective_like(int logI) const
    {
        return !(inc_step >> logI);
    }

    CADO_GPU_FN bool is_vertical_line() const { return inc_step == UINT64_MAX; }
};

/* }}} */
/* {{{ one factor base root, from the entry to the enumerator */

/* How fill_in_buckets_toplevel() would treat a given lattice. The
 * ordinary code has four branches and we mirror them one for one; the
 * split matters because bucket updates have to stay grouped by slice,
 * so anything we hand back to the host must be something that does not
 * write into the per-slice update stream.
 */
enum gpu_fib_kind : int {
    /* the common case: walk the lattice, one update per coprime hit */
    GPU_FIB_GENERIC = 0,
    /* root 0: a single update, no coprimality test */
    GPU_FIB_VERTICAL = 1,
    /* nothing to do at all */
    GPU_FIB_DISCARDED = 2,
    /* projective-like, or p dividing the special-q. The host replays
     * the ordinary code for these. It can do so without disturbing the
     * slice boundaries because a projective-like lattice contributes
     * row updates, which live in their own per-bucket vectors and carry
     * their own slice index; and because p | q cannot actually happen,
     * the host having filtered it out with is_coprime_to() first. */
    GPU_FIB_HOST = 3
};

/* Mirror of the Q.is_coprime_to(e.p) guard. A prime that divides the
 * special-q is not sieved: its lattice would be degenerate.
 */
CADO_GPU_FN bool gpu_fib_entry_is_sieved(gpu_fib_plan const & plan, uint32_t p)
{
    for (int k = 0; k < plan.n_q_factors; k++)
        if (plan.q_factors[k] == p)
            return false;
    return true;
}

/* Build the enumerator for one (entry, root) pair and say which branch
 * it falls in. Mirrors the body of the loop in
 * fill_in_buckets_toplevel(), up to but excluding the walk.
 */
CADO_GPU_FN int gpu_fib_prepare(gpu_ple * ple, gpu_fib_entry const & e,
                                gpu_qlattice const & basis,
                                gpu_fib_fence const & F, int logI)
{
    uint32_t r;
    int const kind = gpu_fb_root_in_qlattice(&r, e.p, e.r, e.invq, basis);
    if (kind < 0)
        return GPU_FIB_HOST;

    gpu_plattice pli;
    pli.initial_basis(e.p, r, kind == 1);
    pli.reduce(UINT32_C(1) << logI);

    ple->init(pli, logI);
    /* skip (i,j) = (0,0), exactly as the host does before testing
     * is_discarded() */
    ple->next(F);

    if (pli.is_discarded())
        return GPU_FIB_DISCARDED;
    /* the host tests projective-like first, then vertical; the two are
     * not mutually exclusive, so the order is part of the contract */
    if (ple->is_projective_like(logI))
        return GPU_FIB_HOST;
    if (ple->is_vertical_line())
        return GPU_FIB_VERTICAL;
    return GPU_FIB_GENERIC;
}

/* Enumerate the positions that the ordinary code would push, calling
 * emit(x) for each, in increasing x. Used twice per pass: once to count
 * and once to scatter, which is cheaper than remembering the positions.
 */
template <typename EMIT>
CADO_GPU_FN void gpu_fib_walk(gpu_ple ple, int kind, gpu_fib_fence const & F,
                              EMIT emit)
{
    if (kind == GPU_FIB_GENERIC) {
        while (!ple.done(F)) {
            if (ple.probably_coprime(F))
                emit(ple.x);
            ple.next(F);
        }
    } else if (kind == GPU_FIB_VERTICAL) {
        /* note that the ordinary code does not test coprimality here */
        if (!ple.done(F))
            emit(ple.x);
    }
}

/* }}} */

#endif /* CADO_LAS_GPU_FIB_KERNELS_HPP */
