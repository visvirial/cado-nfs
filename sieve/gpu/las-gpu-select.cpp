#include "cado.h" // IWYU pragma: keep

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

#include "fmt/format.h"

#include "las-gpu-select.hpp"

namespace las_gpu {

namespace {
/* Written by select(), before any thread starts -- or, once, by
 * suspended(), when a device that had no memory for las at first gets
 * some: hence the atomic copy that everybody reads. */
std::unique_ptr<gpu_engine> owned_engine;
std::atomic<gpu_engine *> the_engine {nullptr};
/* The CUDA device to open once it has memory to spare, or -1. */
int pending_device = -1;
/* The CUDA device behind the_engine, or -1. */
std::atomic<int> opened_device {-1};
std::atomic<unsigned> the_stages {STAGE_BUCKETS | STAGE_ECM};

/* Offloading is suspended, not stopped, when the device fails us: the
 * usual cause is memory that another process holds for a while. So it
 * is retried after a while, which doubles with every failure in a row
 * and goes back to the minimum once a pass gets through. Times are
 * steady_clock ticks; 0 means not suspended. */
using suspend_clock = std::chrono::steady_clock;
constexpr suspend_clock::duration suspend_min = std::chrono::seconds(30);
constexpr suspend_clock::duration suspend_max = std::chrono::minutes(30);
std::atomic<suspend_clock::rep> suspended_until {0};
std::atomic<suspend_clock::rep> suspend_length {suspend_min.count()};
}

static void set_engine(gpu_engine * e)
{
    owned_engine.reset(e);
    the_engine = e;
}

void select(std::string const & spec)
{
    set_engine(nullptr);
    pending_device = -1;
    opened_device = -1;

    if (spec.empty() || spec == "none")
        return;

    if (spec == "cpu") {
        set_engine(gpu_engine_cpu_create());
        return;
    }

    int device = 0;
    bool want_cuda = false;
    bool fallback = false;

    if (spec == "auto") {
        want_cuda = true;
        fallback = true;
    } else if (spec == "cuda") {
        want_cuda = true;
    } else if (spec.compare(0, 5, "cuda:") == 0) {
        want_cuda = true;
        try {
            device = std::stoi(spec.substr(5));
        } catch (std::exception const &) {
            throw std::runtime_error(
                fmt::format("las-gpu: cannot parse device number in \"{}\"",
                            spec));
        }
    } else {
        throw std::runtime_error(
            fmt::format("las-gpu: unknown backend \"{}\""
                        " (expected none, cpu, cuda, cuda:<n> or auto)",
                        spec));
    }

    if (want_cuda) {
        bool oom = false;
        gpu_engine * e = gpu_engine_cuda_create(device, &oom);
        if (e) {
            set_engine(e);
            opened_device = device;
            return;
        }
        if (oom) {
            /* Not for good: whoever holds the memory may give it back. */
            pending_device = device;
            suspend(fmt::format("CUDA device {} has no memory free for"
                                " las", device).c_str());
            return;
        }
        if (!fallback) {
            throw std::runtime_error(fmt::format(
                "las-gpu: no usable CUDA device {} ({})", device,
                gpu_engine_cuda_compiled()
                    ? "no such device, or driver unavailable"
                    : "cado-nfs was built without CUDA support"));
        }
        /* Not the cpu backend: it is a reference implementation, and
         * much slower than not offloading at all. */
    }
}

void select_stages(std::string const & spec)
{
    if (spec == "none") {
        the_stages = 0;
        return;
    }
    if (spec == "all") {
        the_stages = STAGE_NORMS | STAGE_SURVIVORS | STAGE_BUCKETS | STAGE_ECM;
        return;
    }

    unsigned s = 0;
    for (size_t i = 0; i < spec.size();) {
        size_t const j = spec.find(',', i);
        std::string const name = spec.substr(i, j - i);
        if (name == "norms")
            s |= STAGE_NORMS;
        else if (name == "survivors")
            s |= STAGE_SURVIVORS;
        else if (name == "buckets")
            s |= STAGE_BUCKETS;
        else if (name == "ecm")
            s |= STAGE_ECM;
        else
            throw std::runtime_error(fmt::format(
                "las-gpu: unknown stage \"{}\""
                " (expected norms, survivors, buckets, ecm, all or none)",
                name));
        if (j == std::string::npos)
            break;
        i = j + 1;
    }
    the_stages = s;
}

static suspend_clock::rep now()
{
    return suspend_clock::now().time_since_epoch().count();
}

static bool suspended()
{
    suspend_clock::rep until = suspended_until.load(std::memory_order_relaxed);
    if (!until)
        return false;
    if (now() < until)
        return true;
    /* Whoever ends the suspension is the one to open the device, if it
     * has not been yet; the others go on without it meanwhile. */
    if (!suspended_until.compare_exchange_strong(until, 0))
        return false;
    if (!the_engine.load() && pending_device >= 0) {
        gpu_engine * e = gpu_engine_cuda_create(pending_device);
        if (!e) {
            suspend(fmt::format("CUDA device {} still has no memory free"
                                " for las", pending_device).c_str());
            return true;
        }
        set_engine(e);
        opened_device = pending_device;
        fmt::print(stderr, "# las-gpu: offload backend: {}\n", e->name());
    }
    fmt::print(stderr, "# las-gpu: offloading resumes\n");
    return false;
}

bool stage_selected(stage s)
{
    return (the_engine.load() || pending_device >= 0) &&
        (the_stages.load(std::memory_order_relaxed) & unsigned(s));
}

bool stage_enabled(stage s)
{
    return stage_selected(s) && !suspended() && the_engine.load();
}

void suspend(char const * why)
{
    suspend_clock::rep const t = now();
    suspend_clock::rep const length = suspend_length.load();
    suspend_clock::rep until = suspended_until.load();
    /* Several threads may fail at once over the same shortage: only
     * the first one to get here starts a suspension. One that has
     * run out, but that nobody has noticed yet, does not count. */
    do {
        if (until > t)
            return;
    } while (!suspended_until.compare_exchange_weak(until, t + length));
    suspend_length = std::min(2 * length, suspend_max.count());
    fmt::print(stderr,
            "# las-gpu: {}; offloading is suspended for {} s, and the"
            " sieving carries on on the CPU meanwhile\n", why,
            std::chrono::duration_cast<std::chrono::seconds>(
                suspend_clock::duration(length)).count());
}

void pass_succeeded()
{
    if (suspend_length.load(std::memory_order_relaxed) != suspend_min.count())
        suspend_length = suspend_min.count();
}

int cuda_device()
{
    return opened_device;
}

gpu_engine * engine()
{
    return the_engine.load();
}

std::string describe()
{
    if (gpu_engine * e = the_engine.load())
        return e->name();
    if (pending_device >= 0)
        return fmt::format("cuda:{} (once it has memory free)",
                           pending_device);
    return "none";
}

} // namespace las_gpu
