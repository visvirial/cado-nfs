#include "cado.h" // IWYU pragma: keep

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <gmp.h>

#include "cxx_mpz.hpp"
#include "ecm/facul.hpp"
#include "ecm/facul_ecm.h"
#include "ecm/facul_method.hpp"
#include "ecm/facul_stepper.hpp"
#include "ecm/modset.hpp"
#include "gpu/las-gpu-ecm.hpp"
#include "gpu/las-gpu-select.hpp"
#include "las-auxiliary-data.hpp"
#include "las-cofac-standalone.hpp"
#include "las-detached-cofac.hpp"
#include "las-gpu-cofac.hpp"
#include "las-info.hpp"
#include "las-threads-work-data.hpp"
#include "tdict.hpp"
#include "threadpool.hpp"
#include "utils_cxx.hpp"

#ifndef HAVE_CUDA
std::unique_ptr<gpu_ecm_runner> gpu_ecm_runner_create(int)
{
    return nullptr;
}
#endif

namespace {

/* Below this many numbers for a curve, the CPU does it. The latency of a
 * device pass is that of a whole curve, a few milliseconds whatever the
 * size of the batch, and a thread does a curve in less than 100us. */
constexpr size_t MIN_DEVICE_BATCH = 128;

/* numbers per task, at least */
constexpr size_t MIN_CHUNK = 256;

gpu_ecm_runner * get_runner()
{
    static std::once_flag once;
    static std::unique_ptr<gpu_ecm_runner> runner;
    std::call_once(once, []() {
        int const device = las_gpu::cuda_device();
        if (device >= 0)
            runner = gpu_ecm_runner_create(device);
    });
    return runner.get();
}

bool device_wants(facul_method const & method, FaculModulusBase const & n)
{
    if (method.method != EC_METHOD)
        return false;
    auto const & plan = *static_cast<ecm_plan_t const *>(method.plan);
    if (plan.parameterization != MONTY12 || !plan.bc)
        return false;
    if (plan.B1 >= plan.stage2.B2)
        return false;
    cxx_mpz const z = n.get_z();
    return mpz_odd_p(z) && mpz_sizeinbase(z, 2) <= 126;
}

/* A batch of survivors, going through their strategies. There is no
 * thread in charge of it, which would have to wait for the others: the
 * last task of a round starts the next one. */
class cofac_run : public std::enable_shared_from_this<cofac_run> {
    struct item {
        cofac_standalone cur;
        std::vector<int> cof_bitsize;
        std::unique_ptr<facul_stepper> stepper;
        uint64_t n[2] = { 0, 0 };

        explicit item(cofac_standalone && c) : cur(std::move(c)) {}
    };

    /* in increasing order of B1, which is the order in which the default
     * strategy has them. The order has no bearing on the outcome. */
    using curve = std::pair<unsigned int, ecm_plan_t const *>;

    std::shared_ptr<nfs_work_cofac> wc_p;
    std::shared_ptr<nfs_aux> aux_p;
    std::vector<std::unique_ptr<item>> items;

    std::mutex mu;
    std::map<curve, std::vector<item *>> waiting;
    curve last_curve { 0, nullptr };

    /* the round in progress; set up with no task running */
    enum class step { start, supply, run_on_cpu };
    step current_step = step::start;
    std::vector<item *> current;
    std::vector<gpu_ecm_result> results;
    std::atomic<size_t> remaining { 0 };

    void launch_round(thread_pool & pool)
    {
        size_t const n = current.size();
        size_t const nchunks = std::clamp<size_t>(n / MIN_CHUNK, 1,
                                                  4 * pool.size());
        remaining = nchunks;
        for (size_t c = 0; c < nchunks; c++) {
            pool.add_task(thread_pool::QUEUE_ECM,
                    [self = shared_from_this(),
                     lo = n * c / nchunks,
                     hi = n * (c + 1) / nchunks](worker_thread * worker) {
                        self->one_task(worker, lo, hi);
                    });
        }
    }

    void one_task(worker_thread * worker, size_t const lo, size_t const hi)
    {
        nfs_aux & aux(*aux_p);
        timetree_t & timer(aux.get_timer(worker));
        ENTER_THREAD_TIMER(timer);
        auto tt = worker->trace(chronograms::ECM());

        SIBLING_TIMER(timer, "cofactoring");
        TIMER_CATEGORY(timer, cofactoring_mixed());

        std::map<curve, std::vector<item *>> next;

        for (size_t i = lo; i < hi; i++) {
            item & it(*current[i]);
            switch (current_step) {
                case step::start:
                    it.cof_bitsize.assign(wc_p->las.cpoly.nsides(), 0);
                    wc_p->las.cofac_stats.call(it.cur.norm, it.cof_bitsize);
                    it.stepper = std::make_unique<facul_stepper>(
                            it.cur.norm, *wc_p->strategies);
                    break;
                case step::supply: {
                    cxx_mpz f;
                    mpz_import(f, 2, -1, sizeof(uint64_t), 0, 0, results[i].f);
                    it.stepper->supply(f);
                    break;
                }
                case step::run_on_cpu:
                    it.stepper->run_pending();
                    break;
            }

            if (it.stepper->advance(device_wants)) {
                cxx_mpz const z = it.stepper->pending_modulus().get_z();
                it.n[0] = mpz_getlimbn(z, 0);
                it.n[1] = mpz_size(z) > 1 ? mpz_getlimbn(z, 1) : 0;
                auto const * plan = static_cast<ecm_plan_t const *>(
                        it.stepper->pending_method().plan);
                next[curve(plan->B1, plan)].push_back(&it);
            } else {
                int const pass = it.cur.conclude_leftover_norms(
                        it.stepper->result());
                detached_cofac_conclude(worker, *wc_p, aux, it.cur, pass,
                        it.cof_bitsize);
                it.stepper.reset();
                SIBLING_TIMER(timer, "cofactoring");
                TIMER_CATEGORY(timer, cofactoring_mixed());
            }
        }

        {
            std::lock_guard<std::mutex> const lock(mu);
            for (auto & kv: next) {
                auto & w = waiting[kv.first];
                w.insert(w.end(), kv.second.begin(), kv.second.end());
            }
        }

        if (remaining.fetch_sub(1) == 1)
            next_round(worker);
    }

    /* From the last task of a round: no other is running. */
    void next_round(worker_thread * worker)
    {
        if (waiting.empty())
            return;

        /* Sweep through the curves. The numbers that finish with a curve
         * then go along with the others to the next one, and those that
         * are done with one side of a relation gather up for the next
         * sweep. */
        auto it = waiting.upper_bound(last_curve);
        if (it == waiting.end())
            it = waiting.begin();
        last_curve = it->first;
        current = std::move(it->second);
        waiting.erase(it);

        size_t const n = current.size();
        current_step = step::run_on_cpu;

        if (n >= MIN_DEVICE_BATCH) {
            timetree_t & timer(aux_p->get_timer(worker));
            SIBLING_TIMER(timer, "cofactoring on the device");
            TIMER_CATEGORY(timer, cofactoring_mixed());

            std::vector<uint64_t> moduli(2 * n);
            for (size_t i = 0; i < n; i++) {
                moduli[2 * i] = current[i]->n[0];
                moduli[2 * i + 1] = current[i]->n[1];
            }
            results.resize(n);
            if (get_runner()->run(*last_curve.second, moduli.data(), n,
                                  results.data())) {
                current_step = step::supply;
            } else {
                static std::once_flag warned;
                std::call_once(warned, []() {
                    fprintf(stderr, "# las-gpu: no room on the device for"
                            " ECM; the CPU does the curves that do not fit\n");
                });
            }
        }

        launch_round(worker->get_pool());
    }

    public:
    cofac_run(std::shared_ptr<nfs_work_cofac> const & wc_p,
              std::shared_ptr<nfs_aux> const & aux_p,
              std::vector<cofac_standalone> && survivors)
        : wc_p(wc_p), aux_p(aux_p)
    {
        items.reserve(survivors.size());
        current.reserve(survivors.size());
        for (auto & s: survivors) {
            items.emplace_back(std::make_unique<item>(std::move(s)));
            current.push_back(items.back().get());
        }
    }

    void start(thread_pool & pool)
    {
        if (!current.empty())
            launch_round(pool);
    }
};

} // namespace

std::shared_ptr<las_gpu_cofac> las_gpu_cofac::create()
{
    if (!las_gpu::stage_enabled(las_gpu::STAGE_ECM) || !get_runner())
        return nullptr;
    return std::make_shared<las_gpu_cofac>();
}

void las_gpu_cofac::add(cofac_standalone && cur)
{
    std::lock_guard<std::mutex> const lock(mu);
    survivors.push_back(std::move(cur));
}

void las_gpu_cofac::flush(std::shared_ptr<nfs_work_cofac> const & wc_p,
                          std::shared_ptr<nfs_aux> const & aux_p,
                          thread_pool & pool)
{
    std::vector<cofac_standalone> taken;
    {
        std::lock_guard<std::mutex> const lock(mu);
        std::swap(taken, survivors);
    }
    std::make_shared<cofac_run>(wc_p, aux_p, std::move(taken))->start(pool);
}
