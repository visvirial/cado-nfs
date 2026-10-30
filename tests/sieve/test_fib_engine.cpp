#include "cado.h" // IWYU pragma: keep

/* Checks that the CUDA bucket-sieving backend produces exactly what the
 * host one does.
 *
 * tests/sieve/test_fib_gpu already pins the arithmetic the two share;
 * what is left to go wrong is the plumbing that only the device has --
 * the counting pass, the per-(slice, bucket) layout, the atomics in the
 * scatter, and the way the updates are copied back. This test exercises
 * exactly that, on a synthetic factor base: the engines do not care
 * whether the primes come from a real polynomial, only that they are
 * odd and at least I.
 *
 * Updates within one (slice, bucket) pair are compared as a multiset,
 * because the device scatters with atomics and therefore does not
 * preserve the order the host walks in. Nothing downstream depends on
 * that order -- apply_one_bucket adds logp at each position, and
 * relations are sorted before they are printed -- but the multiset must
 * match exactly, one missing update being one missing relation.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <vector>

#include <gmp.h>

#include "las-arith.hpp"
#include "macros.h"
#include "tests_common.h"

#include "gpu/las-gpu-engine.hpp"
#include "gpu/las-gpu-fib-plan.hpp"

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

/* The two shapes that need a conspiracy between the root and the
 * q-lattice, and that random roots essentially never produce:
 *
 *   r == a0/b0 mod p   makes the transformed root projective, which is
 *                      what the device hands back to the host
 *   r == a1/b1 mod p   makes it zero, i.e. a vertical line, which the
 *                      device does handle but by a separate branch
 *
 * Both matter: the first is the only path by which an entry escapes the
 * device at all, and the second contributes to the per-slice update
 * stream without going through the ordinary walk.
 */
static bool special_root(uint32_t * out, uint32_t p, int64_t num, int64_t den)
{
    cxx_mpz mp, mn, md;
    mpz_set_ui(mp, p);
    mpz_set_int64(mn, num);
    mpz_set_int64(md, den);
    mpz_mod(mn, mn, mp);
    mpz_mod(md, md, mp);
    if (!mpz_invert(md, md, mp))
        return false;
    mpz_mul(mn, mn, md);
    mpz_mod(mn, mn, mp);
    *out = uint32_t(mpz_get_ui(mn));
    return true;
}

static int64_t pick(gmp_randstate_t rs, int bits)
{
    cxx_mpz z;
    mpz_urandomb(z, rs, unsigned(bits));
    int64_t v = int64_t(mpz_get_ui(z));
    return gmp_urandomm_ui(rs, 2) ? -v : v;
}

namespace {
struct run_output {
    std::vector<uint32_t> counts;
    std::vector<uint32_t> offsets;
    std::vector<gpu_fib_update> updates;
    std::vector<uint32_t> host_entries;
};
}

static void collect(gpu_engine * eng, gpu_fib_pass * pass, run_output & out)
{
    gpu_fib_output const o = eng->fib_wait(pass);
    size_t const grid = size_t(o.nslices) * o.n_buckets;
    out.counts.assign(o.counts, o.counts + grid);
    out.offsets.assign(o.offsets, o.offsets + grid + 1);
    out.updates.assign(o.updates, o.updates + o.offsets[grid]);
    out.host_entries.assign(o.host_entries, o.host_entries + o.nhost);
    eng->fib_release(pass);
}

static bool compare(run_output const & a, run_output const & b,
                    uint32_t nslices, unsigned long & nupdates)
{
    if (a.host_entries.size() != b.host_entries.size()) {
        fprintf(stderr, "FAIL: %zu vs %zu entries handed back\n",
                a.host_entries.size(), b.host_entries.size());
        return false;
    }
    {
        std::vector<uint32_t> ha = a.host_entries;
        std::vector<uint32_t> hb = b.host_entries;
        std::ranges::sort(ha);
        std::ranges::sort(hb);
        if (ha != hb) {
            fprintf(stderr, "FAIL: different entries handed back\n");
            return false;
        }
    }
    for (size_t i = 0; i < a.counts.size(); i++)
        if (a.counts[i] != b.counts[i] || a.offsets[i] != b.offsets[i]) {
            fprintf(stderr,
                    "FAIL: bucket %zu slice %zu: %" PRIu32 "@%" PRIu32
                    " vs %" PRIu32 "@%" PRIu32 " updates\n",
                    i / nslices, i % nslices, a.counts[i], a.offsets[i],
                    b.counts[i], b.offsets[i]);
            return false;
        }
    if (a.offsets.back() != b.offsets.back()) {
        fprintf(stderr, "FAIL: different totals\n");
        return false;
    }

    /* same layout: now the contents, (bucket, slice) by (bucket,
     * slice), as multisets */
    auto key = [](gpu_fib_update const & u) {
        return (uint32_t(u.x) << 16) | u.hint;
    };
    for (size_t i = 0; i < a.counts.size(); i++) {
        uint32_t const o = a.offsets[i];
        uint32_t const n = a.counts[i];
        std::vector<uint32_t> va(n), vb(n);
        for (uint32_t k = 0; k < n; k++) {
            va[k] = key(a.updates[o + k]);
            vb[k] = key(b.updates[o + k]);
        }
        std::ranges::sort(va);
        std::ranges::sort(vb);
        if (va != vb) {
            fprintf(stderr,
                    "FAIL: bucket %zu slice %zu (%" PRIu32
                    " updates) differs\n", i / nslices, i % nslices, n);
            return false;
        }
        nupdates += n;
    }
    return true;
}

int main(int argc, char const * argv[])
{
    tests_common_cmdline(&argc, &argv, PARSE_SEED | PARSE_ITER);

    unsigned long iter = 10;
    tests_common_get_iter(&iter);

    gpu_engine * const cpu = gpu_engine_cpu_create();
    gpu_engine * const cuda = gpu_engine_cuda_create(0);
    if (!cuda) {
        printf("no CUDA device, nothing to compare against\n");
        delete cpu;
        return EXIT_SUCCESS;
    }

    gmp_randstate_t & rs = state;
    unsigned long nupdates = 0, nhost_total = 0;

    for (unsigned long it = 0; it < iter; it++) {
        int const logI = 11 + int(gmp_urandomm_ui(rs, 4));
        uint32_t const J = 512 + uint32_t(gmp_urandomm_ui(rs, 1536));
        int const logB = 16;

        uint64_t const area = uint64_t(J) << logI;
        uint32_t const nb = uint32_t((area + (1u << logB) - 1) >> logB);

        /* a few thousand lattices spread over a handful of slices */
        size_t const nentries = 2000 + size_t(gmp_urandomm_ui(rs, 6000));
        size_t const nslices = 3 + size_t(gmp_urandomm_ui(rs, 6));

        gpu_qlattice const basis {pick(rs, 31), pick(rs, 31), pick(rs, 31),
                                  pick(rs, 31)};

        std::vector<gpu_fib_entry> entries;
        std::vector<gpu_fib_slice> slices;
        entries.reserve(nentries);
        for (size_t s = 0; s < nslices; s++) {
            gpu_fib_slice sl {};
            sl.first = uint32_t(entries.size());
            sl.slice_index = uint32_t(s);
            size_t const want = (nentries * (s + 1)) / nslices - entries.size();
            for (size_t k = 0; k < want; k++) {
                int const pbits = (logI + 1) +
                    int(gmp_urandomm_ui(rs, unsigned(28 - logI)));
                uint32_t const p = random_odd_prime(rs, pbits, 1u << logI);
                uint32_t r = uint32_t(gmp_urandomm_ui(rs, p));
                /* about one entry in thirty of each special shape */
                unsigned long const dice = gmp_urandomm_ui(rs, 30);
                if (dice == 0)
                    special_root(&r, p, basis.a0, basis.b0);
                else if (dice == 1)
                    special_root(&r, p, basis.a1, basis.b1);
                entries.push_back(gpu_fib_entry {
                        p, uint32_t(-invmod_po2(p)), r, uint16_t(k), 0});
            }
            sl.count = uint32_t(entries.size()) - sl.first;
            slices.push_back(sl);
        }

        gpu_fib_plan plan {};
        plan.logI = logI;
        plan.log_bucket_region = logB;
        plan.J = J;
        plan.n_buckets = nb;
        plan.chunk_begin = 0;
        plan.chunk_count = nb;
        plan.basis = basis;
        /* a special-q with one prime factor, the usual case; the
         * kernel has to skip any entry that matches it */
        plan.n_q_factors = 1;
        plan.q_factors[0] = entries[gmp_urandomm_ui(rs, entries.size())].p;

        gpu_fib_task * const tc = cpu->fib_task_create(
                entries.data(), uint32_t(entries.size()), slices.data(),
                uint32_t(slices.size()));
        gpu_fib_task * const tg = cuda->fib_task_create(
                entries.data(), uint32_t(entries.size()), slices.data(),
                uint32_t(slices.size()));

        /* Two device passes in flight at once, as when both sides are
         * sieved concurrently: they must not share scratch. The second
         * one is over a different q-lattice. */
        gpu_fib_plan plan2 = plan;
        plan2.basis = gpu_qlattice {pick(rs, 31), pick(rs, 31),
                                    pick(rs, 31), pick(rs, 31)};

        gpu_fib_pass * const p1 = cuda->fib_launch(tg, plan, nullptr);
        gpu_fib_pass * const p2 = cuda->fib_launch(tg, plan2, nullptr);

        run_output A, B, A2, B2;
        collect(cpu, cpu->fib_launch(tc, plan, nullptr), A);
        collect(cpu, cpu->fib_launch(tc, plan2, nullptr), A2);
        collect(cuda, p2, B2);
        collect(cuda, p1, B);

        nhost_total += (unsigned long) (A.host_entries.size() +
                                        A2.host_entries.size());
        if (!compare(A, B, uint32_t(nslices), nupdates) ||
            !compare(A2, B2, uint32_t(nslices), nupdates)) {
            fprintf(stderr, "  logI=%d J=%" PRIu32 " nb=%" PRIu32
                            " nentries=%zu nslices=%zu\n",
                    logI, J, nb, nentries, nslices);
            exit(EXIT_FAILURE);
        }

        /* The same area in chunks, as las does it for large ones. The
         * device resumes each walk where the previous chunk left it;
         * the host reference starts over every time and keeps what
         * falls in the chunk, so the two get there independently. The
         * last chunk may be short. */
        uint32_t const K = std::max<uint32_t>(1,
                nb / (2 + uint32_t(gmp_urandomm_ui(rs, 4))));
        gpu_fib_state * const st = cuda->fib_state_create(tg);
        for (uint32_t c0 = 0; c0 < nb; c0 += K) {
            gpu_fib_plan pc = plan;
            pc.chunk_begin = c0;
            pc.chunk_count = std::min(K, nb - c0);
            run_output C, D;
            collect(cuda, cuda->fib_launch(tg, pc, st), D);
            collect(cpu, cpu->fib_launch(tc, pc, nullptr), C);
            if (!compare(C, D, uint32_t(nslices), nupdates)) {
                fprintf(stderr, "  logI=%d J=%" PRIu32 " nb=%" PRIu32
                                " nentries=%zu nslices=%zu,"
                                " chunk [%" PRIu32 ", +%" PRIu32 ")\n",
                        logI, J, nb, nentries, nslices, c0,
                        pc.chunk_count);
                exit(EXIT_FAILURE);
            }
        }
        cuda->fib_state_destroy(st);

        cuda->fib_task_destroy(tg);
        cpu->fib_task_destroy(tc);
    }

    delete cuda;
    delete cpu;
    tests_common_clear();

    printf("%lu runs agree, whole and chunked, %lu updates compared,"
           " %lu handed back\n",
           iter, nupdates, nhost_total);
    return EXIT_SUCCESS;
}
