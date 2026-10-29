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
    std::vector<gpu_fib_update> updates;
    std::vector<uint32_t> counts;
    std::vector<uint32_t> host_entries;
    uint64_t nhost = 0;
    bool overflow = false;
};
}

static void run_one(gpu_engine * eng, std::vector<gpu_fib_entry> const & entries,
                    std::vector<gpu_fib_slice> const & slices,
                    gpu_fib_plan const & plan,
                    std::vector<uint64_t> const & begin,
                    std::vector<uint64_t> const & room, size_t storage,
                    run_output & out)
{
    gpu_fib_task * const task = eng->fib_task_create(
            entries.data(), uint32_t(entries.size()), slices.data(),
            uint32_t(slices.size()));

    out.updates.assign(storage, gpu_fib_update {0xffff, 0xffff});
    out.counts.assign(slices.size() * plan.n_buckets, 0);
    out.host_entries.assign(std::max<size_t>(entries.size(), 1), 0);

    gpu_fib_result const r = eng->fib_run(task, plan, begin.data(), room.data(),
                                          out.updates.data(),
                                          out.counts.data(),
                                          out.host_entries.data(),
                                          out.host_entries.size());
    out.nhost = r.nhost;
    out.overflow = r.overflow;
    eng->fib_task_destroy(task);
}

static bool compare(run_output const & a, run_output const & b,
                    std::vector<gpu_fib_slice> const & slices,
                    std::vector<uint64_t> const & begin, uint32_t nb,
                    unsigned long & nupdates)
{
    size_t const nslices = slices.size();

    if (a.overflow != b.overflow) {
        fprintf(stderr, "FAIL: overflow %d vs %d\n", a.overflow, b.overflow);
        return false;
    }
    if (a.nhost != b.nhost) {
        fprintf(stderr, "FAIL: %lu vs %lu entries handed back\n",
                (unsigned long) a.nhost, (unsigned long) b.nhost);
        return false;
    }
    {
        std::vector<uint32_t> ha(a.host_entries.begin(),
                                 a.host_entries.begin() + ptrdiff_t(a.nhost));
        std::vector<uint32_t> hb(b.host_entries.begin(),
                                 b.host_entries.begin() + ptrdiff_t(b.nhost));
        std::ranges::sort(ha);
        std::ranges::sort(hb);
        if (ha != hb) {
            fprintf(stderr, "FAIL: different entries handed back\n");
            return false;
        }
    }
    if (a.counts != b.counts) {
        for (size_t s = 0; s < nslices; s++)
            for (uint32_t b2 = 0; b2 < nb; b2++)
                if (a.counts[s * nb + b2] != b.counts[s * nb + b2]) {
                    fprintf(stderr,
                            "FAIL: slice %zu bucket %" PRIu32
                            ": %" PRIu32 " vs %" PRIu32 " updates\n",
                            s, b2, a.counts[s * nb + b2],
                            b.counts[s * nb + b2]);
                    return false;
                }
    }

    /* same counts: now the contents, bucket by bucket and slice by
     * slice, as multisets */
    auto key = [](gpu_fib_update const & u) {
        return (uint32_t(u.x) << 16) | u.hint;
    };
    for (uint32_t b2 = 0; b2 < nb; b2++) {
        uint64_t off = begin[b2];
        for (size_t s = 0; s < nslices; s++) {
            uint32_t const n = a.counts[s * nb + b2];
            std::vector<uint32_t> va(n), vb(n);
            for (uint32_t k = 0; k < n; k++) {
                va[k] = key(a.updates[off + k]);
                vb[k] = key(b.updates[off + k]);
            }
            std::ranges::sort(va);
            std::ranges::sort(vb);
            if (va != vb) {
                fprintf(stderr,
                        "FAIL: bucket %" PRIu32 " slice %zu (%" PRIu32
                        " updates) differs\n", b2, s, n);
                return false;
            }
            nupdates += n;
            off += n;
        }
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
        plan.basis = basis;
        /* a special-q with one prime factor, the usual case; the
         * kernel has to skip any entry that matches it */
        plan.n_q_factors = 1;
        plan.q_factors[0] = entries[gmp_urandomm_ui(rs, entries.size())].p;

        /* A lattice of determinant p hits the area about area/p
         * times; give each bucket twice its share of that, and grow if
         * the estimate was unlucky. We want no overflow here: the
         * overflow path is las's business, and it exercises it on its
         * own whenever a bucket turns out too small. */
        uint64_t est = 0;
        for (auto const & e: entries)
            est += area / e.p;
        size_t per_bucket = size_t(2 * est / nb) + 1024;

        std::vector<uint64_t> begin(nb), room(nb);
        run_output A, B;
        for (;;) {
            size_t const gap = 7;
            for (uint32_t b = 0; b < nb; b++) {
                begin[b] = uint64_t(b) * (per_bucket + gap) + gap;
                room[b] = per_bucket;
            }
            run_one(cpu, entries, slices, plan, begin, room,
                    size_t(nb) * (per_bucket + gap) + gap, A);
            if (!A.overflow)
                break;
            per_bucket *= 2;
        }
        run_one(cuda, entries, slices, plan, begin, room,
                size_t(nb) * (per_bucket + 7) + 7, B);
        nhost_total += (unsigned long) A.nhost;
        if (!compare(A, B, slices, begin, nb, nupdates)) {
            fprintf(stderr, "  logI=%d J=%" PRIu32 " nb=%" PRIu32
                            " nentries=%zu nslices=%zu\n",
                    logI, J, nb, nentries, nslices);
            exit(EXIT_FAILURE);
        }
    }

    delete cuda;
    delete cpu;
    tests_common_clear();

    printf("%lu passes agree, %lu updates compared, %lu handed back\n",
           iter, nupdates, nhost_total);
    return EXIT_SUCCESS;
}
