#include "cado.h" // IWYU pragma: keep

/* Checks that the device-portable bucket-sieving arithmetic of
 * sieve/gpu/las-gpu-fib-kernels.hpp agrees with the ordinary sieve code,
 * for the whole chain: root transform, p-lattice reduction, and the
 * Franke-Kleinjung walk.
 *
 * "Agrees" here means exactly: the two must produce the same
 * classification (generic / discarded / special) and, for the generic
 * ones, the very same sequence of x positions. One missing update is one
 * missing relation.
 *
 * This is a pure-arithmetic test; it does not need a device, and it is
 * what tells us that a mismatch found later in the pipeline is a
 * bookkeeping problem rather than a numerical one.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <vector>

#include <gmp.h>

#include "las-arith.hpp"
#include "las-fbroot-qlattice.hpp"
#include "las-plattice.hpp"
#include "las-qlattice.hpp"
#include "macros.h"
#include "tests_common.h"

#include "gpu/las-gpu-fib-kernels.hpp"
#include "gpu/las-gpu-fib-plan.hpp"

/* Bucket sieving only ever sees primes above bucket_thresh, which is at
 * least I; reduce_plattice() has p >= I as a precondition and asserts on
 * it. Sampling below that would be testing a case las cannot produce.
 */
static uint32_t random_odd_prime(gmp_randstate_t rs, int bits, uint32_t pmin)
{
    cxx_mpz z;
    for (;;) {
        mpz_urandomb(z, rs, unsigned(bits));
        mpz_setbit(z, bits - 1);
        mpz_nextprime(z, z);
        if (mpz_fits_uint_p(z)) {
            uint32_t const p = uint32_t(mpz_get_ui(z));
            if (p >= pmin)
                return p;
        }
    }
}

/* -1/p mod 2^32, the same invq that the factor base stores. */
static redc_invp_t compute_invq(fbprime_t p)
{
    return redc_invp_t(-invmod_po2(p));
}

static void random_basis(qlattice_basis & basis, gmp_randstate_t rs, int bits)
{
    auto pick = [&]() {
        cxx_mpz z;
        mpz_urandomb(z, rs, unsigned(bits));
        int64_t v = int64_t(mpz_get_ui(z));
        if (gmp_urandomm_ui(rs, 2))
            v = -v;
        return v;
    };
    basis.a0 = pick();
    basis.a1 = pick();
    basis.b0 = pick();
    basis.b1 = pick();
}

/* The reference: exactly what fill_in_buckets_toplevel() does for one
 * (entry, root) pair, with the walk recorded instead of pushed.
 */
static int reference_walk(std::vector<uint64_t> & out, fbprime_t p,
                          fbroot_t r, redc_invp_t invq,
                          qlattice_basis const & basis, int logI, uint32_t J)
{
    out.clear();

    fb_root_p1 const R = fb_root_in_qlattice(p, fb_root_p1::affine_root(r),
                                             invq, basis);
    plattice_info const pli(p, R.r, R.proj, logI);
    plattice_enumerator ple(pli, 0, logI);
    plattice_enumerator::fence const F(logI, int(J));

    ple.next(F);

    if (pli.is_discarded())
        return GPU_FIB_DISCARDED;
    if (ple.is_projective_like(logI))
        return GPU_FIB_HOST;
    if (pli.is_vertical_line(logI)) {
        if (!ple.done(F))
            out.push_back(ple.get_x());
        return GPU_FIB_VERTICAL;
    }

    while (!ple.done(F)) {
        if (ple.probably_coprime(F))
            out.push_back(ple.get_x());
        ple.next(F);
    }
    return GPU_FIB_GENERIC;
}

static int portable_walk(std::vector<uint64_t> & out, fbprime_t p, fbroot_t r,
                         redc_invp_t invq, qlattice_basis const & basis,
                         int logI, uint32_t J)
{
    out.clear();

    gpu_qlattice qb {basis.a0, basis.b0, basis.a1, basis.b1};
    gpu_fib_entry const e {p, invq, r, 0, 0};
    gpu_fib_fence const F = gpu_fib_fence_for(logI, J);

    gpu_ple ple {};
    int const kind = gpu_fib_prepare(&ple, e, qb, F, logI);
    gpu_fib_walk(ple, kind, F, [&](uint64_t x) { out.push_back(x); });
    return kind;
}

int main(int argc, char const * argv[])
{
    tests_common_cmdline(&argc, &argv, PARSE_SEED | PARSE_ITER);

    unsigned long iter = 200000;
    tests_common_get_iter(&iter);

    /* tests_common_cmdline() seeds the shared `state` from -seed. */
    gmp_randstate_t & rs = state;

    std::vector<uint64_t> a, b;
    unsigned long nhost = 0, nvertical = 0, ndiscarded = 0, nupdates = 0;

    for (unsigned long it = 0; it < iter; it++) {
        /* A spread of sizes: logI from 10 to 16 is the range las
         * actually uses, and p must be sampled over the whole
         * bucket-sieved range because the reduction behaves quite
         * differently for p below and above I. */
        int const logI = 10 + int(gmp_urandomm_ui(rs, 7));
        uint32_t const J = 1 + uint32_t(gmp_urandomm_ui(rs, 1u << 12));
        /* p must be at least I, and we want the whole range above it:
         * the reduction behaves quite differently for p just above I and
         * for p near the factor base bound. */
        int const pbits = (logI + 1) + int(gmp_urandomm_ui(rs,
                                       unsigned(31 - (logI + 1))));

        /* The 31-bit root transform requires a basis that fits; that is
         * a precondition las checks in choose_sieve_area(). */
        qlattice_basis basis;
        do {
            random_basis(basis, rs, 31);
        } while (!basis.fits_31bits());

        fbprime_t const p = random_odd_prime(rs, pbits, 1u << logI);
        redc_invp_t const invq = compute_invq(p);
        fbroot_t const r = fbroot_t(gmp_urandomm_ui(rs, p));

        int const ka = reference_walk(a, p, r, invq, basis, logI, J);
        int const kb = portable_walk(b, p, r, invq, basis, logI, J);

        if (ka != kb) {
            fprintf(stderr,
                    "FAIL: classification differs for p=%" PRIu32
                    " r=%" PRIu32 " logI=%d J=%" PRIu32
                    ": reference %d, portable %d\n",
                    p, r, logI, J, ka, kb);
            fprintf(stderr, "  basis = (%" PRId64 ", %" PRId64 ", %" PRId64
                            ", %" PRId64 ")\n",
                    basis.a0, basis.b0, basis.a1, basis.b1);
            exit(EXIT_FAILURE);
        }

        if (ka == GPU_FIB_HOST) { nhost++; continue; }
        if (ka == GPU_FIB_DISCARDED) { ndiscarded++; continue; }
        if (ka == GPU_FIB_VERTICAL) nvertical++;

        if (a != b) {
            fprintf(stderr,
                    "FAIL: walk differs for p=%" PRIu32 " r=%" PRIu32
                    " logI=%d J=%" PRIu32 ": %zu vs %zu positions\n",
                    p, r, logI, J, a.size(), b.size());
            size_t const n = std::min(a.size(), b.size());
            for (size_t k = 0; k < n; k++) {
                if (a[k] != b[k]) {
                    fprintf(stderr, "  first difference at %zu: %" PRIu64
                                    " vs %" PRIu64 "\n", k, a[k], b[k]);
                    break;
                }
            }
            exit(EXIT_FAILURE);
        }
        nupdates += a.size();
    }

    tests_common_clear();

    printf("%lu lattices: %lu walked (%lu updates), of which %lu vertical;"
           " %lu left to the host, %lu discarded\n",
           iter, iter - nhost - ndiscarded, nupdates, nvertical, nhost,
           ndiscarded);
    return EXIT_SUCCESS;
}
