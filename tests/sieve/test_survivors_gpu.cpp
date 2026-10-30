#include "cado.h" // IWYU pragma: keep

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <vector>

#include "las-config.hpp"
#include "las-unsieve.hpp"
#include "macros.h"
#include "params.hpp"
#include "gpu/las-gpu-engine.hpp"
#include "gpu/las-gpu-plan.hpp"
#include "gpu/las-gpu-select.hpp"

/* Checks the offloaded survivor search against the one in
 * las-unsieve.cpp, on synthetic sieve arrays.
 *
 * The contract we are verifying is the one that
 * process_bucket_region_run::search_survivors() relies on:
 *   - the same survivors, in the same (increasing) order,
 *   - the same 255 markers left behind in the Sf array, which
 *     purge_buckets(), resieve() and the cofactoring then read.
 *
 * Synthetic arrays are the right input here: what we are testing is the
 * coprimality bookkeeping and the compaction, neither of which cares
 * where the norms came from, and random arrays let us reach survivor
 * densities that a real sieve would take hours to produce.
 */

namespace {

struct rng {
    uint64_t s;
    explicit rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1) {}
    uint32_t next()
    {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return uint32_t(s >> 32);
    }
    uint32_t below(uint32_t n) { return next() % n; }
};

/* Verbatim transcription of the loop in
 * process_bucket_region_run::search_survivors(). */
std::vector<uint32_t> reference_search(unsigned char * S0, unsigned char * S1,
                                       unsigned char const bound[2],
                                       int logI, unsigned int N,
                                       j_divisibility_helper const & jd,
                                       unsigned int td_max,
                                       unsieve_data const & us,
                                       sublat_runtime_t sublat)
{
    std::vector<uint32_t> sv;

    unsigned int const log_lines_per_region = MAX(0, LOG_BUCKET_REGION - logI);
    unsigned int const log_regions_per_line = MAX(0, logI - LOG_BUCKET_REGION);
    unsigned int const regions_per_line = 1U << log_regions_per_line;
    unsigned int const region_rank_in_line = N & (regions_per_line - 1);
    unsigned int const j0 = (N >> log_regions_per_line) << log_lines_per_region;
    unsigned int const j1 = j0 + (1U << log_lines_per_region);
    int const I = 1 << logI;
    int const i0 = int(region_rank_in_line << LOG_BUCKET_REGION) - I / 2;
    int const i1 = i0 + (1 << MIN(LOG_BUCKET_REGION, logI));

    for (unsigned int j = j0; j < j1; j++) {
        int const offset = int(j - j0) << logI;
        unsigned char * const both_S[2] = {
            S0 ? S0 + offset : nullptr,
            S1 ? S1 + offset : nullptr,
        };
        unsigned char const both_bounds[2] = { bound[0], bound[1] };
        size_t const old_size = sv.size();
        search_survivors_in_line(both_S, both_bounds, j, i0, i1, int(N), jd,
                                 td_max, us, sv, sublat);
        if (!offset)
            continue;
        for (size_t k = old_size; k < sv.size(); k++)
            sv[k] += unsigned(offset);
    }
    return sv;
}

int failures = 0;

void report(char const * what, std::string const & ctx)
{
    fprintf(stderr, "FAILED [%s]: %s\n", ctx.c_str(), what);
    failures++;
}

void one_case(gpu_engine * eng, gpu_survivor_task * task,
              j_divisibility_helper const & jd, unsieve_data const & us,
              int logI, int logA, unsigned int N, int nsides, int drop_side,
              unsigned char b0, unsigned char b1, unsigned int td_max,
              sublat_runtime_t sublat, uint64_t seed)
{
    size_t const region_size = size_t(1) << LOG_BUCKET_REGION;

    char buf[256];
    snprintf(buf, sizeof(buf),
             "logI=%d logA=%d B=%d N=%u nsides=%d drop=%d b=(%u,%u) td=%u "
             "sublat=(%u,%u,%u) seed=%lu",
             logI, logA, LOG_BUCKET_REGION, N, nsides, drop_side, b0, b1,
             td_max, sublat.m, sublat.i0, sublat.j0, (unsigned long) seed);
    std::string const ctx(buf);

    /* Values are drawn from a range that straddles the bounds, so that
     * both branches of the test get exercised with a high density. */
    rng r(seed);
    std::vector<unsigned char> base0(region_size), base1(region_size);
    for (size_t x = 0; x < region_size; x++) {
        base0[x] = (unsigned char) r.below(2u * b0 + 2u);
        base1[x] = (unsigned char) r.below(2u * b1 + 2u);
    }

    std::vector<unsigned char> ref0(base0), ref1(base1);
    std::vector<unsigned char> gpu0(base0), gpu1(base1);

    bool const has0 = !(nsides == 1 && drop_side == 0);
    bool const has1 = !(nsides == 1 && drop_side == 1);

    unsigned char const bounds[2] = { b0, b1 };
    auto const ref = reference_search(has0 ? ref0.data() : nullptr,
                                      has1 ? ref1.data() : nullptr, bounds,
                                      logI, N, jd, td_max, us, sublat);

    gpu_survivor_region R {};
    R.nsides = nsides;
    R.side_of_Sf = has0 ? 0 : 1;
    R.bound[0] = b0;
    R.bound[1] = b1;
    R.logI = logI;
    R.td_max = td_max;
    /* the kernel's "no sublattice" is m == 0 */
    R.sublat_m = sublat.m == 1 ? 0 : sublat.m;
    R.sublat_i0 = sublat.i0;
    R.sublat_j0 = sublat.j0;
    {
        unsigned int const llpr = MAX(0, LOG_BUCKET_REGION - logI);
        unsigned int const lrpl = MAX(0, logI - LOG_BUCKET_REGION);
        unsigned int const rpl = 1U << lrpl;
        unsigned int const rank = N & (rpl - 1);
        R.geo.j0 = (N >> lrpl) << llpr;
        R.geo.j1 = R.geo.j0 + (1U << llpr);
        R.geo.I = 1 << logI;
        R.geo.i0 = int(rank << LOG_BUCKET_REGION) - R.geo.I / 2;
        R.geo.i1 = R.geo.i0 + (1 << MIN(LOG_BUCKET_REGION, logI));
        R.geo.has_origin = (R.geo.j0 == 0) && (rank == (rpl - 1) / 2);
    }

    std::vector<uint32_t> out(region_size);
    size_t const n = eng->survivor_search(
        task, has0 ? gpu0.data() : nullptr, has1 ? gpu1.data() : nullptr,
        region_size, R, out.data(), out.size());

    /* Report the first place where the two lists diverge before
     * reporting the count: when they differ, the offending (i,j) is what
     * one wants to see, and the count alone says nothing. */
    size_t const ncommon = MIN(n, ref.size());
    for (size_t k = 0; k < ncommon; k++) {
        if (out[k] != ref[k]) {
            uint32_t const x = MIN(out[k], ref[k]);
            int const line_len = R.geo.i1 - R.geo.i0;
            snprintf(buf, sizeof(buf),
                     "survivor %zu: reference x=%u, gpu x=%u"
                     " (first differs at x=%u, i=%d, j=%u)",
                     k, ref[k], out[k], x,
                     R.geo.i0 + int(x % unsigned(line_len)),
                     R.geo.j0 + x / unsigned(line_len));
            report(buf, ctx);
            return;
        }
    }
    if (n != ref.size()) {
        /* One list is a prefix of the other; the extra entry is the
         * interesting one. */
        uint32_t const x = n > ref.size() ? out[ncommon] : ref[ncommon];
        int const line_len = R.geo.i1 - R.geo.i0;
        snprintf(buf, sizeof(buf),
                 "survivor count: reference %zu, gpu %zu"
                 " (first extra at x=%u, i=%d, j=%u)",
                 ref.size(), n, x, R.geo.i0 + int(x % unsigned(line_len)),
                 R.geo.j0 + x / unsigned(line_len));
        report(buf, ctx);
        return;
    }

    std::vector<unsigned char> const & rf = has0 ? ref0 : ref1;
    std::vector<unsigned char> const & gf = has0 ? gpu0 : gpu1;
    for (size_t x = 0; x < region_size; x++) {
        if (rf[x] != gf[x]) {
            snprintf(buf, sizeof(buf),
                     "Sf[%zu]: reference %u, gpu %u", x, rf[x], gf[x]);
            report(buf, ctx);
            return;
        }
    }
}

} // namespace

// coverity[root_function]
int main(int argc, char const * argv[])
{
    cxx_param_list pl;
    pl.declare_usage("gpu", "backend to test (cpu, cuda, cuda:<n>, auto)");
    pl.declare_usage("seed", "base seed for the synthetic sieve arrays");
    pl.declare_usage("log-bucket-region", "set bucket region to 2^x");
    pl.configure_alias("log-bucket-region", "B");
    pl.process_command_line(argc, argv, false);

    std::string backend = "cpu";
    pl.parse("gpu", backend);
    unsigned long seed = 1;
    pl.parse("seed", seed);
    pl.parse("log-bucket-region", LOG_BUCKET_REGION);
    set_LOG_BUCKET_REGION();

    las_gpu::select(backend);
    gpu_engine * const eng = las_gpu::engine();
    ASSERT_ALWAYS(eng != nullptr);
    printf("# testing survivor search on backend %s, B=2^%d\n", eng->name(),
           LOG_BUCKET_REGION);

    int const logA = 22;
    /* logI must stay in a range where J = 2^(logA-logI) is at least 2,
     * which is what j_divisibility_helper wants. */
    int const logIs[] = { 10, 12, 16, 17 };

    unsigned long nb_cases = 0;

    for (int const logI : logIs) {
        if (logI > logA - 1)
            continue;
        uint32_t const J = 1U << (logA - logI);
        j_divisibility_helper const jd(J);
        unsieve_data const us(logI, logA);

        std::vector<gpu_jdiv_entry> jtab(J);
        for (uint32_t k = 0; k < J; k++)
            jtab[k] = { jd[int(k)].p, jd[int(k)].cof, jd[int(k)].inv,
                        jd[int(k)].bound };

        gpu_survivor_task * const task = eng->survivor_task_create(
            jtab.data(), J, size_t(1) << LOG_BUCKET_REGION);

        uint64_t const nregions =
            (uint64_t(1) << (logA - LOG_BUCKET_REGION));

        rng pick(seed * 97 + unsigned(logI));

        for (int rep = 0; rep < 24; rep++) {
            /* region 0 carries the j = 0 line, which is special-cased;
             * make sure we always look at it. */
            unsigned int const N =
                rep < 2 ? unsigned(rep)
                        : unsigned(pick.next() % nregions);
            unsigned char const b0 = (unsigned char) (20 + pick.below(120));
            unsigned char const b1 = (unsigned char) (20 + pick.below(120));
            unsigned int const td_max =
                (unsigned int[]){ 2, 3, 7, 13, 1024 }[pick.below(5)];

            for (int nsides = 2; nsides >= 1; nsides--) {
                for (int drop = 0; drop < (nsides == 1 ? 2 : 1); drop++) {
                    /* No sublattices: las does not offload the survivor
                     * search under them (see search_survivors()). */
                    one_case(eng, task, jd, us, logI, logA, N, nsides, drop,
                             b0, b1, td_max, sublat_runtime_t(),
                             seed + uint64_t(nb_cases));
                    nb_cases++;
                }
            }
        }

        eng->survivor_task_destroy(task);
    }

    printf("# %lu cases, %d failures\n", nb_cases, failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
