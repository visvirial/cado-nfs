/* ECM on the device against ecm<layer>() on the host.
 *
 * For every MONTY12 curve of the default cofactorization strategy -- the
 * one las uses -- and for moduli shaped like las's cofactors, the device
 * must find the same factor, with the same backtracking flag, as the CPU
 * code does. See sieve/gpu/las-gpu-ecm.hpp.
 *
 * With -time, it also says how long both take.
 */

#include "cado.h" // IWYU pragma: keep

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <chrono>
#include <vector>

#include <gmp.h>

#include "arithxx/modredc126.hpp"
#include "arithxx/modredc64.hpp"
#include "arithxx/modredc96.hpp"
#include "cxx_mpz.hpp"
#include "ecm.hpp"
#include "facul_ecm.h"
#include "facul.hpp"
#include "facul_method.hpp"
#include "facul_stepper.hpp"
#include "facul_strategies.hpp"
#include "gpu/las-gpu-ecm.hpp"
#include "macros.h"
#include "tests_common.h"

namespace {

struct outcome {
    uint64_t f[2];
    int bt;
    bool operator==(outcome const & o) const
    {
        return f[0] == o.f[0] && f[1] == o.f[1] && bt == o.bt;
    }
};

/* What facul_doit_onefm would run: the smallest layer that fits. */
outcome cpu_ecm(uint64_t const n[2], ecm_plan_t const & plan)
{
    outcome r {};
    Integer128 const N(n[0], n[1]);
    size_t const bits = N.bits();
    if (bits <= 64) {
        arithxx_modredc64::Modulus const m { Integer64(n[0]) };
        Integer64 f;
        r.bt = ecm<arithxx_modredc64>(f, m, plan);
        r.f[0] = f.getWord(0);
    } else if (bits <= 96) {
        arithxx_modredc96::Modulus const m(N);
        Integer128 f;
        r.bt = ecm<arithxx_modredc96>(f, m, plan);
        r.f[0] = f.getWord(0);
        r.f[1] = f.getWord(1);
    } else {
        arithxx_modredc126::Modulus const m(N);
        Integer128 f;
        r.bt = ecm<arithxx_modredc126>(f, m, plan);
        r.f[0] = f.getWord(0);
        r.f[1] = f.getWord(1);
    }
    return r;
}

void random_prime(cxx_mpz & p, gmp_randstate_t rs, unsigned int bits)
{
    do {
        mpz_urandomb(p, rs, bits - 1);
        mpz_setbit(p, bits - 1);
        mpz_nextprime(p, p);
    } while (mpz_sizeinbase(p, 2) != bits);
}

/* Cofactors as las meets them: products of primes above the factor base
 * bound, most of them too large, a few with a smooth split. */
void random_cofactor(uint64_t n[2], gmp_randstate_t rs, unsigned int kind)
{
    cxx_mpz N, p;
    mpz_set_ui(N, 1);
    auto mulprime = [&](unsigned int bits) {
        random_prime(p, rs, bits);
        mpz_mul(N, N, p);
    };
    switch (kind % 7) {
        case 0: /* three large primes */
            mulprime(31 + gmp_urandomm_ui(rs, 9));
            mulprime(31 + gmp_urandomm_ui(rs, 9));
            mulprime(31 + gmp_urandomm_ui(rs, 9));
            break;
        case 1: /* one large prime and a composite that is too large */
            mulprime(31 + gmp_urandomm_ui(rs, 9));
            mulprime(40 + gmp_urandomm_ui(rs, 40));
            break;
        case 2: /* two primes, the size of las's two-large-prime case */
            mulprime(31 + gmp_urandomm_ui(rs, 9));
            mulprime(31 + gmp_urandomm_ui(rs, 9));
            break;
        case 3: /* a small factor, which ECM finds easily */
            mulprime(20 + gmp_urandomm_ui(rs, 10));
            mulprime(60 + gmp_urandomm_ui(rs, 30));
            break;
        case 4: /* one 64-bit-sized number, for the other layer */
            mulprime(20 + gmp_urandomm_ui(rs, 12));
            mulprime(20 + gmp_urandomm_ui(rs, 12));
            break;
        case 5: /* as large as the modulus may be */
            mulprime(39);
            mulprime(39);
            mulprime(48);
            break;
        default: /* a prime power, which tends to give back n itself */
            mulprime(31 + gmp_urandomm_ui(rs, 9));
            mpz_mul(N, N, N);
            mulprime(31 + gmp_urandomm_ui(rs, 9));
            break;
    }
    ASSERT_ALWAYS(mpz_sizeinbase(N, 2) <= 126);
    n[0] = mpz_getlimbn(N, 0);
    n[1] = mpz_size(N) > 1 ? mpz_getlimbn(N, 1) : 0;
}

double seconds_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
        .count();
}

} // namespace

int main(int argc, char const * argv[])
{
    unsigned long count = 3000;

    tests_common_cmdline(&argc, &argv, PARSE_SEED | PARSE_ITER | PARSE_TIME);
    tests_common_get_iter(&count);
    int const bench = tests_common_get_time();

    /* The c270 parameters, which are where the cofactorization costs
     * most; the curves only depend on ncurves anyway. */
    facul_strategies const strat({ 1071225238, 2146644785 }, { 38, 39 },
                                 { 109, 120 }, { 28, 31 }, true, 0);
    std::vector<ecm_plan_t const *> eplans;
    for (auto const & ms: strat({ 109, 120 })) {
        if (ms.method->method != EC_METHOD)
            continue;
        auto const * plan = static_cast<ecm_plan_t const *>(ms.method->plan);
        if (plan->parameterization != MONTY12)
            continue;
        bool seen = false;
        for (auto const * q: eplans)
            seen = seen || q == plan;
        if (!seen)
            eplans.push_back(plan);
    }
    ASSERT_ALWAYS(!eplans.empty());

    auto runner = gpu_ecm_runner_create(0);
    if (!runner) {
        printf("# no CUDA device, nothing to test\n");
        return 0;
    }

    std::vector<uint64_t> n(2 * count);
    for (unsigned long i = 0; i < count; i++)
        random_cofactor(&n[2 * i], state, (unsigned int) i);

    std::vector<gpu_ecm_result> got(count);
    unsigned long failures = 0, found = 0, backtracked = 0, whole = 0;
    double cpu_time = 0, gpu_time = 0;

    /* once to get the device warm, as las would be */
    ASSERT_ALWAYS(runner->run(*eplans[0], n.data(), count, got.data()));

    for (size_t p = 0; p < eplans.size(); p++) {
        auto t0 = std::chrono::steady_clock::now();
        ASSERT_ALWAYS(runner->run(*eplans[p], n.data(), count, got.data()));
        gpu_time += seconds_since(t0);

        t0 = std::chrono::steady_clock::now();
        std::vector<outcome> want(count);
        for (unsigned long i = 0; i < count; i++)
            want[i] = cpu_ecm(&n[2 * i], *eplans[p]);
        cpu_time += seconds_since(t0);

        for (unsigned long i = 0; i < count; i++) {
            outcome const g { { got[i].f[0], got[i].f[1] }, got[i].bt };
            if (!(g == want[i])) {
                if (failures++ < 10)
                    fprintf(stderr,
                            "FAILED: B1=%u param=%u n=0x%016lx%016lx:"
                            " cpu f=0x%lx%016lx bt=%d,"
                            " gpu f=0x%lx%016lx bt=%d\n",
                            eplans[p]->B1, (unsigned int) eplans[p]->parameter,
                            (unsigned long) n[2 * i + 1],
                            (unsigned long) n[2 * i],
                            (unsigned long) want[i].f[1],
                            (unsigned long) want[i].f[0], want[i].bt,
                            (unsigned long) g.f[1], (unsigned long) g.f[0],
                            g.bt);
            }
            if (want[i].f[0] != 1 || want[i].f[1] != 0)
                found++;
            if (want[i].f[0] == n[2 * i] && want[i].f[1] == n[2 * i + 1])
                whole++;
            if (want[i].bt)
                backtracked++;
        }
    }

    /* The whole strategy, with the device doing what it can, one number
     * at a time, must come to what facul_all() comes to. */
    {
        unsigned long const npairs = std::min(count, 2000UL);
        auto wants = [](facul_method const & m, FaculModulusBase const & z) {
            if (m.method != EC_METHOD)
                return false;
            auto const * plan = static_cast<ecm_plan_t const *>(m.plan);
            return plan->parameterization == MONTY12 &&
                mpz_sizeinbase(z.get_z(), 2) <= 126;
        };
        bool (*const want)(facul_method const &, FaculModulusBase const &) =
            wants;
        unsigned long offloaded = 0;
        for (unsigned long i = 0; i + 1 < npairs; i += 2) {
            std::vector<cxx_mpz> N(2);
            mpz_import(N[0], 2, -1, sizeof(uint64_t), 0, 0, &n[2 * i]);
            mpz_import(N[1], 2, -1, sizeof(uint64_t), 0, 0, &n[2 * i + 2]);
            auto const want_res = facul_all(N, strat);
            facul_stepper S(N, strat);
            while (S.advance(want)) {
                cxx_mpz const z = S.pending_modulus().get_z();
                uint64_t const w[2] = {
                    mpz_getlimbn(z, 0),
                    mpz_size(z) > 1 ? mpz_getlimbn(z, 1) : 0
                };
                gpu_ecm_result r;
                auto const * plan = static_cast<ecm_plan_t const *>(
                        S.pending_method().plan);
                ASSERT_ALWAYS(runner->run(*plan, w, 1, &r));
                cxx_mpz f;
                mpz_import(f, 2, -1, sizeof(uint64_t), 0, 0, r.f);
                S.supply(f);
                offloaded++;
            }
            auto const & got_res = S.result();
            bool ok = true;
            for (int side = 0; side < 2; side++)
                ok = ok && got_res[side].status == want_res[side].status
                        && got_res[side].primes == want_res[side].primes;
            if (!ok && failures++ < 10)
                gmp_fprintf(stderr, "FAILED: strategy on %Zd, %Zd\n",
                            (mpz_srcptr) N[0], (mpz_srcptr) N[1]);
        }
        printf("# whole strategy on %lu pairs: %lu curves on the device\n",
               npairs / 2, offloaded);
    }

    unsigned long const curves = count * eplans.size();
    printf("# %zu MONTY12 curves (B1 %u..%u) x %lu moduli: %lu factors"
           " found (%lu of them n itself), %lu backtracked, %lu mismatches\n",
           eplans.size(), eplans.front()->B1, eplans.back()->B1, count, found,
           whole, backtracked, failures);
    if (bench)
        printf("# cpu: %.2f us/curve (one thread); gpu: %.3f us/curve;"
               " ratio %.1f\n",
               1e6 * cpu_time / double(curves),
               1e6 * gpu_time / double(curves), cpu_time / gpu_time);

    tests_common_clear();
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
