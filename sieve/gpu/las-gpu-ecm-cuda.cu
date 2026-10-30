/* ECM on the device, MONTY12 curves: see las-gpu-ecm.hpp.
 *
 * Every function below is a transcription of its namesake in
 * sieve/ecm/ecm.cpp, ec_arith_Montgomery.hpp, ec_parameterization.hpp or
 * ec_arith_Weierstrass_new.cpp, with the same operations in the same
 * order. Keep them that way: the outcome only matches the CPU's because
 * the residues do, and the residues only do because the operations do.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "las-gpu-ecm.hpp"

#define ECM_CUDA_CHECK(call)                                                  \
    do {                                                                      \
        cudaError_t const e_ = (call);                                        \
        if (e_ != cudaSuccess) {                                              \
            fprintf(stderr, "%s:%d: CUDA error in %s: %s\n", __FILE__,        \
                    __LINE__, #call, cudaGetErrorString(e_));                 \
            abort();                                                          \
        }                                                                     \
    } while (0)

namespace {

using u128 = unsigned __int128;

/* {{{ arithmetic modulo n < 2^126, in Montgomery form with R = 2^128 */

struct dmod {
    u128 n;
    uint64_t ninv;      /* -1/n mod 2^64 */
    u128 r2;            /* R^2 mod n */
};

__device__ __forceinline__ uint64_t lo(u128 x) { return (uint64_t) x; }
__device__ __forceinline__ uint64_t hi(u128 x) { return (uint64_t) (x >> 64); }

/* Montgomery multiplication. Writing this as 32-bit CIOS with PTX carry
 * chains was tried, and is no faster than what the compiler makes of
 * this. */
__device__ __forceinline__ u128 mod_mul(u128 a, u128 b, dmod const & m)
{
    uint64_t const a0 = lo(a), a1 = hi(a), b0 = lo(b), b1 = hi(b);
    uint64_t const n0 = lo(m.n), n1 = hi(m.n);
    uint64_t t[4];
    u128 c;

    u128 const p00 = (u128) a0 * b0;
    u128 const p01 = (u128) a0 * b1;
    u128 const p10 = (u128) a1 * b0;
    u128 const p11 = (u128) a1 * b1;
    t[0] = lo(p00);
    c = (u128) hi(p00) + lo(p01) + lo(p10);
    t[1] = lo(c);
    c = (u128) hi(c) + hi(p01) + hi(p10) + lo(p11);
    t[2] = lo(c);
    t[3] = hi(c) + hi(p11);

    /* a, b < n < 2^126, so everything below stays under 2^255 */
#pragma unroll
    for (int i = 0; i < 2; i++) {
        uint64_t const q = t[i] * m.ninv;
        c = (u128) q * n0 + t[i];
        c = (u128) q * n1 + t[i + 1] + hi(c);
        t[i + 1] = lo(c);
        uint64_t carry = hi(c);
#pragma unroll
        for (int j = i + 2; j < 4; j++) {
            c = (u128) t[j] + carry;
            t[j] = lo(c);
            carry = hi(c);
        }
    }
    u128 r = ((u128) t[3] << 64) | t[2];
    if (r >= m.n)
        r -= m.n;
    return r;
}

__device__ __forceinline__ u128 mod_add(u128 a, u128 b, dmod const & m)
{
    u128 r = a + b;
    if (r >= m.n)
        r -= m.n;
    return r;
}

__device__ __forceinline__ u128 mod_sub(u128 a, u128 b, dmod const & m)
{
    return a >= b ? a - b : a + m.n - b;
}

__device__ __forceinline__ u128 mod_neg(u128 a, dmod const & m)
{
    return a ? m.n - a : 0;
}

__device__ __forceinline__ u128 to_mont(u128 x, dmod const & m)
{
    return mod_mul(x, m.r2, m);
}

__device__ __forceinline__ u128 from_mont(u128 a, dmod const & m)
{
    return mod_mul(a, 1, m);
}

__device__ __forceinline__ u128 mod_set_ul(uint64_t v, dmod const & m)
{
    return to_mont((u128) v % m.n, m);
}

/* Inverse of the value of a, in Montgomery form. False when there is
 * none, like Modulus::inv(). */
__device__ bool mod_inv(u128 & r, u128 a, dmod const & m)
{
    u128 u = from_mont(a, m), v = m.n;
    u128 x1 = 1, x2 = 0;
    /* invariant: x1 * value == u, x2 * value == v (mod n) */
    for (;;) {
        if (u == 1) {
            r = to_mont(x1, m);
            return true;
        }
        if (v == 1) {
            r = to_mont(x2, m);
            return true;
        }
        if (u == 0 || v == 0)
            return false;
        while (!(u & 1)) {
            u >>= 1;
            x1 = (x1 & 1) ? (x1 + m.n) >> 1 : x1 >> 1;
        }
        while (!(v & 1)) {
            v >>= 1;
            x2 = (x2 & 1) ? (x2 + m.n) >> 1 : x2 >> 1;
        }
        if (u >= v) {
            u -= v;
            x1 = mod_sub(x1, x2, m);
        } else {
            v -= u;
            x2 = mod_sub(x2, x1, m);
        }
    }
}

/* gcd(a, n) where a is the representative of a residue, like
 * Modulus::gcd(): R is a power of two and n is odd, so the factor of R
 * does not matter. */
__device__ u128 mod_gcd(u128 a, dmod const & m)
{
    u128 u = a, v = m.n;
    if (u == 0)
        return v;
    /* v is odd, so the factors of 2 of u do not divide the gcd */
    while (!(u & 1))
        u >>= 1;
    for (;;) {
        /* u and v odd */
        if (u > v) {
            u128 const t = u;
            u = v;
            v = t;
        }
        v -= u;
        if (v == 0)
            return u;
        while (!(v & 1))
            v >>= 1;
    }
}

/* }}} */
/* {{{ curves */

struct pt {
    u128 x, z;
};

__device__ __forceinline__ void pt_swap(pt & a, pt & b)
{
    pt const t = a;
    a = b;
    b = t;
}

/* montgomery_dbl_inl, ELLM_SAFE_ADD == 0.
 *
 * Not inlined, and neither is mont_dadd: they are called from dozens of
 * places, and inlining them makes a kernel that keeps missing the
 * instruction cache. It made stage 1 and stage 2 several times slower. */
__device__ __noinline__ void mont_dbl(pt & Q, pt const & P, u128 b,
                                         dmod const & m)
{
    u128 u, v, w;
    u = mod_add(P.x, P.z, m);
    u = mod_mul(u, u, m);
    v = mod_sub(P.x, P.z, m);
    v = mod_mul(v, v, m);
    u128 const qx = mod_mul(u, v, m);
    w = mod_sub(u, v, m);
    u = mod_mul(w, b, m);
    u = mod_add(u, v, m);
    Q.z = mod_mul(w, u, m);
    Q.x = qx;
}

/* montgomery_dadd_inl, ELLM_SAFE_ADD == 0 */
__device__ __noinline__ void mont_dadd(pt & R, pt const & P, pt const & Q,
                                          pt const & D, dmod const & m)
{
    u128 u, v, w;
    u = mod_sub(P.x, P.z, m);
    v = mod_add(Q.x, Q.z, m);
    u = mod_mul(u, v, m);
    w = mod_add(P.x, P.z, m);
    v = mod_sub(Q.x, Q.z, m);
    v = mod_mul(w, v, m);
    w = mod_add(u, v, m);
    v = mod_sub(u, v, m);
    w = mod_mul(w, w, m);
    v = mod_mul(v, v, m);
    u = D.x;
    u128 const dz = D.z;
    R.x = mod_mul(w, dz, m);
    R.z = mod_mul(u, v, m);
}

/* montgomery_smul_ul */
__device__ void mont_smul_ul(pt & R, pt * Rp1, pt const & P, unsigned long k,
                             u128 b, dmod const & m)
{
    pt T0, T1;

    if (k == 0UL) {
        R.x = R.z = 0;
        if (Rp1)
            *Rp1 = P;
    } else if (k == 1UL) {
        R = P;
        if (Rp1)
            mont_dbl(*Rp1, P, b, m);
    } else if (k == 2UL) {
        if (Rp1) {
            mont_dbl(T1, P, b, m);
            mont_dadd(*Rp1, T1, P, P, m);
            R = T1;
        } else {
            mont_dbl(R, P, b, m);
        }
    } else {
        unsigned long mask = ~(0UL);
        mask -= mask / 2;
        while ((mask & k) == 0)
            mask >>= 1;

        T0 = P;
        mont_dbl(T1, P, b, m);
        mask >>= 1;

        if (!(k & mask)) {
            T0 = T1;
            mont_dadd(T1, T1, P, P, m);
            mask >>= 1;
        }

        for (; mask > 1; mask >>= 1) {
            if (k & mask) {
                mont_dadd(T0, T1, T0, P, m);
                mont_dbl(T1, T1, b, m);
            } else {
                mont_dadd(T1, T1, T0, P, m);
                mont_dbl(T0, T0, b, m);
            }
        }

        if (k & mask) {
            mont_dadd(R, T1, T0, P, m);
            if (Rp1)
                mont_dbl(*Rp1, T1, b, m);
        } else {
            mont_dbl(R, T0, b, m);
            if (Rp1)
                mont_dadd(*Rp1, T1, T0, P, m);
        }
    }
}

/* ECWeierstrass<layer>::AffinePoint, the parts that
 * ec_parameterization_Montgomery12 uses */
struct wpt {
    u128 x, y;
    bool finite;
};

__device__ void w_dbl(wpt & R, wpt const & P, u128 a, dmod const & m)
{
    u128 lambda, u, v;
    if (!P.finite) {
        R.finite = false;
        return;
    }
    u = mod_mul(P.x, P.x, m);
    v = mod_add(u, u, m);
    v = mod_add(v, u, m);
    v = mod_add(v, a, m);
    u = mod_add(P.y, P.y, m);
    bool const ret = mod_inv(lambda, u, m);
    if (ret) {
        lambda = mod_mul(lambda, v, m);
        u = mod_mul(lambda, lambda, m);
        u = mod_sub(u, P.x, m);
        u = mod_sub(u, P.x, m);
        v = mod_sub(P.x, u, m);
        v = mod_mul(v, lambda, m);
        R.y = mod_sub(v, P.y, m);
        R.x = u;
    } else {
        R.x = u;
    }
    R.finite = ret;
}

__device__ bool w_equal(wpt const & P, wpt const & Q)
{
    if (P.finite && Q.finite)
        return P.x == Q.x && P.y == Q.y;
    return P.finite == Q.finite;
}

__device__ void w_add(wpt & R, wpt const & P, wpt const & Q, u128 a,
                      dmod const & m)
{
    u128 lambda, u, v;
    if (!P.finite) {
        R = Q;
        return;
    }
    if (!Q.finite) {
        R = P;
        return;
    }
    u = mod_sub(Q.x, P.x, m);
    bool const ret = mod_inv(v, u, m);
    if (ret) {
        u = mod_sub(Q.y, P.y, m);
        lambda = mod_mul(u, v, m);
        u = mod_mul(lambda, lambda, m);
        u = mod_sub(u, P.x, m);
        u = mod_sub(u, Q.x, m);
        v = mod_sub(P.x, u, m);
        v = mod_mul(v, lambda, m);
        R.y = mod_sub(v, P.y, m);
        R.x = u;
        R.finite = true;
    } else if (w_equal(P, Q)) {
        w_dbl(R, P, a, m);
    } else {
        R.x = u;
        R.finite = false;
    }
}

__device__ void w_smul(wpt & R, wpt const & P, uint64_t e, u128 a,
                       dmod const & m)
{
    if (e == 0) {
        R.finite = false;
    } else if (e == 1) {
        R = P;
    } else {
        uint64_t i;
        wpt T = P;
        i = UINT64_C(1) << 63;
        while ((i & e) == 0)
            i >>= 1;
        i >>= 1;
        for (; i > 0; i >>= 1) {
            w_dbl(T, T, a, m);
            if (e & i)
                w_add(T, T, P, a, m);
        }
        R = T;
    }
}

/* ec_parameterization_Montgomery12 */
__device__ int param_monty12(u128 & b, pt & P0, unsigned long k,
                             dmod const & m)
{
    u128 u, v, a, uu, vv, d1, d2, t1, t2, one, x, y;

    one = mod_set_ul(1, m);
    y = mod_add(one, one, m);
    x = mod_neg(y, m);
    y = mod_add(y, y, m);
    a = mod_add(y, y, m);
    a = mod_add(a, y, m);
    a = mod_neg(a, m);

    wpt T { x, y, true };
    w_smul(T, T, k, a, m);
    if (!T.finite) {
        P0.x = T.x;
        return 0;
    }

    u = T.x;
    v = T.y;
    uu = mod_mul(u, u, m);
    vv = mod_mul(v, v, m);

    t1 = mod_add(uu, uu, m);
    t1 = mod_add(t1, t1, m);
    d2 = mod_sub(vv, t1, m);
    t2 = mod_add(t1, t1, m);
    t2 = mod_add(t2, t1, m);
    d1 = mod_add(vv, t2, m);

    t1 = mod_set_ul(12, m);
    P0.x = mod_add(uu, t1, m);
    P0.x = mod_mul(P0.x, P0.x, m);
    P0.x = mod_mul(P0.x, uu, m);
    P0.x = mod_add(P0.x, P0.x, m);

    P0.z = mod_mul(d1, d2, m);
    P0.z = mod_add(P0.z, P0.z, m);

    t1 = mod_mul(d2, d2, m);
    t1 = mod_mul(t1, d2, m);
    t1 = mod_mul(t1, d1, m);
    bool const ret = mod_inv(b, t1, m);
    if (!ret) {
        P0.x = t1;
    } else {
        t1 = mod_mul(uu, uu, m);
        t1 = mod_mul(t1, uu, m);
        b = mod_mul(b, t1, m);
        b = mod_mul(b, vv, m);
        t1 = mod_set_ul(1024, m);   /* m.pow2(t1, 10) */
        b = mod_mul(b, t1, m);
    }
    return ret;
}

/* bytecode_prac_interpret_montgomery(_internal); the opcodes are
 * bytecode.h's */
__device__ void prac(pt & P, uint8_t const * bc, u128 b, dmod const & m)
{
    pt R[5];
    R[0] = P;
    for (bool finished = false; !finished;) {
        switch (*bc) {
            case PRAC_SWAP:
                pt_swap(R[0], R[1]);
                break;
            case PRAC_SUBBLOCK_INIT:
                R[1] = R[0];
                R[2] = R[0];
                mont_dbl(R[0], R[0], b, m);
                break;
            case PRAC_SUBBLOCK_FINAL:
                mont_dadd(R[0], R[0], R[1], R[2], m);
                break;
            case PRAC_BLOCK_FINAL:
                mont_dadd(R[1], R[0], R[1], R[2], m);
                finished = true;
                break;
            case 1:
                mont_dadd(R[3], R[0], R[1], R[2], m);
                mont_dadd(R[4], R[3], R[0], R[1], m);
                mont_dadd(R[1], R[1], R[3], R[0], m);
                R[0] = R[4];
                break;
            case 2:
                mont_dadd(R[1], R[0], R[1], R[2], m);
                mont_dbl(R[0], R[0], b, m);
                break;
            case 3:
                mont_dadd(R[2], R[1], R[0], R[2], m);
                pt_swap(R[1], R[2]);
                break;
            case 4:
                mont_dadd(R[1], R[1], R[0], R[2], m);
                mont_dbl(R[0], R[0], b, m);
                break;
            case 5:
                mont_dadd(R[2], R[2], R[0], R[1], m);
                mont_dbl(R[0], R[0], b, m);
                break;
            case 6:
                mont_dbl(R[3], R[0], b, m);
                mont_dadd(R[4], R[0], R[1], R[2], m);
                mont_dadd(R[0], R[3], R[0], R[0], m);
                mont_dadd(R[2], R[3], R[4], R[2], m);
                pt_swap(R[1], R[2]);
                break;
            case 7:
                mont_dadd(R[3], R[0], R[1], R[2], m);
                mont_dadd(R[1], R[3], R[0], R[1], m);
                mont_dbl(R[3], R[0], b, m);
                mont_dadd(R[0], R[0], R[3], R[0], m);
                break;
            case 8:
                mont_dadd(R[3], R[0], R[1], R[2], m);
                mont_dadd(R[2], R[2], R[0], R[1], m);
                pt_swap(R[1], R[3]);
                mont_dbl(R[3], R[0], b, m);
                mont_dadd(R[0], R[0], R[3], R[0], m);
                break;
            case 9:
                mont_dadd(R[2], R[2], R[1], R[0], m);
                mont_dbl(R[1], R[1], b, m);
                break;
            case 10:
                mont_dadd(R[1], R[0], R[1], R[2], m);
                R[2] = R[1];
                mont_dbl(R[0], R[1], b, m);
                break;
            case 11:
                mont_dadd(R[2], R[1], R[0], R[2], m);
                pt_swap(R[1], R[2]);
                pt_swap(R[0], R[1]);
                break;
            case 12:
                mont_dadd(R[3], R[1], R[0], R[2], m);
                mont_dadd(R[2], R[0], R[3], R[1], m);
                R[1] = R[2];
                mont_dbl(R[0], R[2], b, m);
                break;
            case 13:
                R[3] = R[1];
                mont_dadd(R[1], R[1], R[0], R[2], m);
                R[2] = R[0];
                mont_dadd(R[0], R[0], R[1], R[3], m);
                break;
            default:
                __trap();
        }
        if (!finished)
            bc++;
    }
    P = R[1];
}

/* }}} */
/* {{{ the plan, on the device */

struct dplan {
    uint32_t parameter, exp2, B1, B2;
    uint32_t w, vmin, vmax, U_len;
    uint8_t const * bc;
    uint32_t const * U;
    uint32_t const * pairs;
};

constexpr uint32_t PAIR_INCR_V_ = UINT32_MAX - 1;
constexpr uint32_t PAIR_END_ = UINT32_MAX;

/* Per-thread scratch for stage 2, strided so that the threads of a warp
 * touch consecutive words: element k of thread t is at k * stride + t. */
struct scratch {
    u128 * base;
    size_t stride;
    __device__ u128 & at(size_t k) const
    {
        return base[k * stride + (blockIdx.x * blockDim.x + threadIdx.x)];
    }
};

/* uP[k] is at x: 2k, z: 2k+1; vwP[k] at x: 2(U_len+k), z: 2(U_len+k)+1;
 * the t[] of common_z after those. */
struct stage2_points {
    scratch s;
    uint32_t U_len, nv;
    __device__ u128 & ux(uint32_t k) const { return s.at(2 * k); }
    __device__ u128 & uz(uint32_t k) const { return s.at(2 * k + 1); }
    __device__ u128 & vx(uint32_t k) const { return s.at(2 * (U_len + k)); }
    __device__ u128 & vz(uint32_t k) const
    {
        return s.at(2 * (U_len + k) + 1);
    }
    __device__ u128 & t(uint32_t k) const
    {
        return s.at(2 * (U_len + nv) + k);
    }
    __device__ pt u(uint32_t k) const { return pt { ux(k), uz(k) }; }
    __device__ pt v(uint32_t k) const { return pt { vx(k), vz(k) }; }
    __device__ void set_u(uint32_t k, pt const & P) const
    {
        ux(k) = P.x;
        uz(k) = P.z;
    }
    __device__ void set_v(uint32_t k, pt const & P) const
    {
        vx(k) = P.x;
        vz(k) = P.z;
    }
};

/* common_z, on the two lists L1 = uP[0..U_len), L2 = vwP[skip..nv) */
__device__ void common_z(stage2_points const & S, uint32_t skip,
                         dmod const & m)
{
    uint32_t const n1 = S.U_len, n2 = S.nv - skip;
    uint32_t const n = n1 + n2;
    if (n <= 1)
        return;

    auto Lx = [&](uint32_t i) -> u128 & {
        return i < n1 ? S.ux(i) : S.vx(skip + i - n1);
    };
    auto Lz = [&](uint32_t i) -> u128 & {
        return i < n1 ? S.uz(i) : S.vz(skip + i - n1);
    };

    u128 p;
    S.t(0) = Lz(0);
    for (uint32_t i = 1; i < n - 1; i++)
        S.t(i) = mod_mul(S.t(i - 1), Lz(i), m);

    Lx(n - 1) = mod_mul(Lx(n - 1), S.t(n - 2), m);
    p = Lz(n - 1);

    for (uint32_t i = n - 2; i > 0; i--) {
        Lx(i) = mod_mul(Lx(i), p, m);
        Lx(i) = mod_mul(Lx(i), S.t(i - 1), m);
        p = mod_mul(p, Lz(i), m);
    }

    Lx(0) = mod_mul(Lx(0), p, m);
}

/* ecm_stage2 */
__device__ int ecm_stage2(u128 & r, pt const & P, dplan const & plan,
                          u128 b, stage2_points const & S, dmod const & m)
{
    pt wP, Pt;
    int bt = 0;

    {
        pt ap1_0, ap1_1, ap5_0, ap5_1, P2, P6;

        ap1_0 = P;
        mont_dbl(P2, P, b, m);
        mont_dadd(P6, P2, P, P, m);
        mont_dadd(ap5_0, P6, P2, P, m);
        mont_dbl(P6, P6, b, m);
        mont_dadd(ap1_1, P6, P, ap5_0, m);
        mont_dadd(ap5_1, P6, ap5_0, P, m);

        uint32_t k = 0;
        if (k < plan.U_len && plan.U[k] == 1)
            S.set_u(k++, ap1_0);
        if (k < plan.U_len && plan.U[k] == 5)
            S.set_u(k++, ap5_0);

        uint32_t u_1mod6 = 7, u_5mod6 = 11;
        while (k < plan.U_len) {
            if (plan.U[k] == u_1mod6) {
                S.set_u(k++, ap1_1);
                continue;
            }
            if (plan.U[k] == u_5mod6) {
                S.set_u(k++, ap5_1);
                continue;
            }

            mont_dadd(Pt, ap1_1, P6, ap1_0, m);
            ap1_0 = ap1_1;
            ap1_1 = Pt;
            u_1mod6 += 6;

            mont_dadd(Pt, ap5_1, P6, ap5_0, m);
            ap5_0 = ap5_1;
            ap5_1 = Pt;
            u_5mod6 += 6;
        }

        if (plan.w == 6) {
            wP = P6;
        } else if (plan.w == 12) {
            mont_dbl(wP, P6, b, m);
        } else if (plan.w % 12 == 0) {
            mont_dadd(Pt, ap1_1, P6, ap1_0, m);
            mont_dadd(wP, Pt, ap5_1, P2, m);
        } else {
            mont_dbl(P2, P2, b, m);
            mont_dadd(wP, ap1_1, ap5_1, P2, m);
        }
    }

    {
        if (plan.vmin == plan.vmax) {
            pt R;
            mont_smul_ul(R, nullptr, wP, plan.vmin, b, m);
            S.set_v(0, R);
        } else {
            pt R0, R1;
            mont_smul_ul(R0, &R1, wP, plan.vmin, b, m);
            S.set_v(0, R0);
            S.set_v(1, R1);
            uint32_t k = 2;
            for (uint32_t v = plan.vmin + 2; v <= plan.vmax; k++, v++) {
                pt R;
                if (v % 2 == 0 && v / 2 >= plan.vmin)
                    mont_dbl(R, S.v(v / 2 - plan.vmin), b, m);
                else
                    mont_dadd(R, S.v(k - 1), wP, S.v(k - 2), m);
                S.set_v(k, R);
            }
        }
    }

    common_z(S, plan.vmin == 0 ? 1 : 0, m);

    {
        u128 t, r_bak;
        r = S.ux(0);
        r_bak = r;

        uint32_t const * u_ptr = plan.pairs;
        for (uint32_t v_idx = 0;; v_idx++, u_ptr++) {
            r_bak = r;
            u128 const vx = S.vx(v_idx);
            for (; *u_ptr != PAIR_END_ && *u_ptr != PAIR_INCR_V_; u_ptr++) {
                t = mod_sub(vx, S.ux(*u_ptr), m);
                r = mod_mul(r, t, m);
            }
            if (r == 0) {
                r = r_bak;
                bt = 1;
                break;
            }
            if (*u_ptr == PAIR_END_)
                break;
        }
    }

    return bt;
}

/* ecm<layer>(), MONTY12 */
__device__ int ecm_monty12(u128 & f, dplan const & plan,
                           stage2_points const & S, dmod const & m)
{
    u128 u, b;
    pt P, Pt;
    int bt = 0;

    f = 1;

    if (param_monty12(b, P, plan.parameter, m) == 0) {
        f = mod_gcd(P.x, m);
        return 0;
    }

    prac(P, plan.bc, b, m);

    Pt = P;
    for (uint32_t i = 0; i < plan.exp2; i++) {
        mont_dbl(P, P, b, m);
        if (P.z == 0) {
            P = Pt;
            bt = 1;
            break;
        }
        Pt = P;
    }
    f = mod_gcd(P.z, m);

    if (bt == 0 && f == 1 && plan.B1 < plan.B2) {
        bt = ecm_stage2(u, P, plan, b, S, m);
        f = mod_gcd(u, m);
    }

    return bt;
}

/* }}} */

/* The per-modulus constants: -1/n mod 2^64 by Newton, and R^2 mod n by
 * doubling 1 up 256 times. n < 2^126, so the doublings do not overflow. */
__global__ void kernel_ecm_prepare(dmod * __restrict__ mods, size_t count)
{
    size_t const i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    u128 const N = mods[i].n;
    uint64_t inv = 1;
    for (int k = 0; k < 6; k++)
        inv *= 2 - (uint64_t) N * inv;
    u128 r = 1 % N;
    for (int k = 0; k < 256; k++) {
        r <<= 1;
        if (r >= N)
            r -= N;
    }
    mods[i].ninv = (uint64_t) -inv;
    mods[i].r2 = r;
}

__global__ void kernel_ecm(dplan plan, dmod const * __restrict__ mods,
                           size_t count, u128 * scratch_base,
                           size_t stride, gpu_ecm_result * out)
{
    size_t const i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    dmod const m = mods[i];
    stage2_points const S { scratch { scratch_base, stride }, plan.U_len,
                            plan.vmax - plan.vmin + 1 };
    u128 f;
    int const bt = ecm_monty12(f, plan, S, m);
    out[i].f[0] = (uint64_t) f;
    out[i].f[1] = (uint64_t) (f >> 64);
    out[i].bt = bt;
    out[i].pad = 0;
}

/* {{{ host side */

/* An ecm_plan_t, on the device. */
struct device_plan {
    uint32_t parameter = 0, exp2 = 0, B1 = 0, B2 = 0;
    uint32_t w = 0, vmin = 0, vmax = 0, U_len = 0;
    uint8_t * bc = nullptr;
    uint32_t * U = nullptr;
    uint32_t * pairs = nullptr;

    /* u128 per thread: the points, and the t[] of common_z */
    size_t scratch_words() const
    {
        size_t const npoints = size_t(U_len) + vmax - vmin + 1;
        return 2 * npoints + npoints;
    }
};

/* false, with the error cleared, when out of memory; anything else is a
 * bug. */
static bool ecm_alloc_ok(cudaError_t const e, char const * what)
{
    if (e == cudaSuccess)
        return true;
    cudaGetLastError();
    if (e == cudaErrorMemoryAllocation)
        return false;
    fprintf(stderr, "%s:%d: CUDA error in %s: %s\n", __FILE__, __LINE__,
            what, cudaGetErrorString(e));
    abort();
}
#define ECM_ALLOC(call) ecm_alloc_ok((call), #call)

class cuda_ecm_runner : public gpu_ecm_runner {
    int device;
    std::mutex mu;
    std::map<ecm_plan_t const *, device_plan> plans;

    dmod * d_mods = nullptr;
    gpu_ecm_result * d_out = nullptr;
    u128 * d_scratch = nullptr;
    size_t capacity = 0;        /* moduli */
    size_t scratch_words = 0;   /* u128 per thread */

    static constexpr unsigned int block = 128;
    /* Beyond a few tens of thousands the device is at full speed, and the
     * scratch is about 5kB per modulus. */
    static constexpr size_t max_batch = size_t(1) << 17;

    static void free_plan(device_plan & dp)
    {
        cudaFree(dp.bc);
        cudaFree(dp.U);
        cudaFree(dp.pairs);
    }

    device_plan const * get_plan(ecm_plan_t const & plan)
    {
        auto it = plans.find(&plan);
        if (it != plans.end())
            return &it->second;

        if (plan.parameterization != MONTY12)
            throw std::invalid_argument("gpu_ecm: MONTY12 only");

        size_t bc_len = 1;
        while (plan.bc[bc_len - 1] != PRAC_BLOCK_FINAL)
            bc_len++;
        size_t pairs_len = 1;
        while (plan.stage2.pairs[pairs_len - 1] != PAIR_END)
            pairs_len++;
        size_t const U_bytes = std::max<size_t>(plan.stage2.U_len, 1)
                               * sizeof(uint32_t);
        size_t const pairs_bytes = pairs_len * sizeof(uint32_t);

        device_plan dp;
        dp.parameter = (uint32_t) plan.parameter;
        dp.exp2 = plan.exp2;
        dp.B1 = plan.B1;
        dp.B2 = plan.stage2.B2;
        dp.w = plan.stage2.w;
        dp.vmin = plan.stage2.vmin;
        dp.vmax = plan.stage2.vmax;
        dp.U_len = plan.stage2.U_len;
        if (!ECM_ALLOC(cudaMalloc(&dp.bc, bc_len)) ||
                !ECM_ALLOC(cudaMalloc(&dp.U, U_bytes)) ||
                !ECM_ALLOC(cudaMalloc(&dp.pairs, pairs_bytes))) {
            free_plan(dp);
            return nullptr;
        }
        ECM_CUDA_CHECK(cudaMemcpy(dp.bc, plan.bc, bc_len,
                                  cudaMemcpyHostToDevice));
        /* the two are unsigned int on the host */
        static_assert(sizeof(unsigned int) == sizeof(uint32_t));
        if (plan.stage2.U_len)
            ECM_CUDA_CHECK(cudaMemcpy(dp.U, plan.stage2.U,
                                      plan.stage2.U_len * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice));
        ECM_CUDA_CHECK(cudaMemcpy(dp.pairs, plan.stage2.pairs, pairs_bytes,
                                  cudaMemcpyHostToDevice));
        return &plans.emplace(&plan, dp).first->second;
    }

    bool reserve(size_t const count, size_t const words)
    {
        if (count <= capacity && words <= scratch_words)
            return true;
        cudaFree(d_mods);
        cudaFree(d_out);
        cudaFree(d_scratch);
        d_mods = nullptr;
        d_out = nullptr;
        d_scratch = nullptr;
        /* the sizes drift up over the first few special-q's */
        size_t const want = std::min(std::max(count + count / 4, capacity),
                                     max_batch);
        capacity = (std::max(want, count) + block - 1) / block * block;
        scratch_words = std::max(words, scratch_words);
        if (!ECM_ALLOC(cudaMalloc(&d_mods, capacity * sizeof(dmod))) ||
                !ECM_ALLOC(cudaMalloc(&d_out,
                                      capacity * sizeof(gpu_ecm_result))) ||
                !ECM_ALLOC(cudaMalloc(&d_scratch, capacity * scratch_words *
                                                      sizeof(u128)))) {
            cudaFree(d_mods);
            cudaFree(d_out);
            cudaFree(d_scratch);
            d_mods = nullptr;
            d_out = nullptr;
            d_scratch = nullptr;
            capacity = 0;
            return false;
        }
        return true;
    }

    void run_one_batch(device_plan const & dp, uint64_t const * n,
                       size_t const count, gpu_ecm_result * out,
                       std::vector<dmod> & mods)
    {
        mods.resize(count);
        for (size_t i = 0; i < count; i++) {
            u128 const N = ((u128) n[2 * i + 1] << 64) | n[2 * i];
            if (!(N & 1) || (N >> 126))
                throw std::invalid_argument("gpu_ecm: modulus must be odd"
                                            " and below 2^126");
            mods[i] = dmod { N, 0, 0 };
        }
        ECM_CUDA_CHECK(cudaMemcpy(d_mods, mods.data(), count * sizeof(dmod),
                                  cudaMemcpyHostToDevice));
        unsigned int const grid = (unsigned int) ((count + block - 1) / block);
        kernel_ecm_prepare<<<grid, block>>>(d_mods, count);
        ECM_CUDA_CHECK(cudaGetLastError());
        dplan const plan { dp.parameter, dp.exp2, dp.B1, dp.B2,
                           dp.w, dp.vmin, dp.vmax, dp.U_len,
                           dp.bc, dp.U, dp.pairs };
        kernel_ecm<<<grid, block>>>(plan, d_mods, count, d_scratch, capacity,
                                    d_out);
        ECM_CUDA_CHECK(cudaGetLastError());
        ECM_CUDA_CHECK(cudaMemcpy(out, d_out, count * sizeof(gpu_ecm_result),
                                  cudaMemcpyDeviceToHost));
    }

public:
    explicit cuda_ecm_runner(int device) : device(device) {}

    ~cuda_ecm_runner() override
    {
        cudaSetDevice(device);
        for (auto & kv: plans)
            free_plan(kv.second);
        cudaFree(d_mods);
        cudaFree(d_out);
        cudaFree(d_scratch);
    }

    bool run(ecm_plan_t const & plan, uint64_t const * n, size_t count,
             gpu_ecm_result * out) override
    {
        std::lock_guard<std::mutex> const lock(mu);
        ECM_CUDA_CHECK(cudaSetDevice(device));
        device_plan const * dp = get_plan(plan);
        if (!dp)
            return false;
        std::vector<dmod> mods;
        for (size_t done = 0; done < count;) {
            size_t const m = std::min(count - done, max_batch);
            if (!reserve(m, dp->scratch_words()))
                return false;
            run_one_batch(*dp, n + 2 * done, m, out + done, mods);
            done += m;
        }
        return true;
    }
};

/* }}} */

} // namespace

std::unique_ptr<gpu_ecm_runner> gpu_ecm_runner_create(int device)
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || device >= count) {
        cudaGetLastError();
        return nullptr;
    }
    return std::make_unique<cuda_ecm_runner>(device);
}
