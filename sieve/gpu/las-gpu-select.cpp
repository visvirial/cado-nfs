#include "cado.h" // IWYU pragma: keep

#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>

#include "fmt/format.h"

#include "las-gpu-select.hpp"

namespace las_gpu {

namespace {
std::unique_ptr<gpu_engine> the_engine;
std::atomic<unsigned> the_stages {STAGE_BUCKETS};
std::atomic<bool> disabled {false};
}

void select(std::string const & spec)
{
    if (spec.empty() || spec == "none") {
        the_engine.reset();
        return;
    }

    if (spec == "cpu") {
        the_engine.reset(gpu_engine_cpu_create());
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
        gpu_engine * e = gpu_engine_cuda_create(device);
        if (e) {
            the_engine.reset(e);
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
        the_engine.reset();
    }
}

void select_stages(std::string const & spec)
{
    if (spec == "none") {
        the_stages = 0;
        return;
    }
    if (spec == "all") {
        the_stages = STAGE_NORMS | STAGE_SURVIVORS | STAGE_BUCKETS;
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
        else
            throw std::runtime_error(fmt::format(
                "las-gpu: unknown stage \"{}\""
                " (expected norms, survivors, buckets, all or none)",
                name));
        if (j == std::string::npos)
            break;
        i = j + 1;
    }
    the_stages = s;
}

bool stage_enabled(stage s)
{
    return the_engine && !disabled.load(std::memory_order_relaxed) &&
        (the_stages.load(std::memory_order_relaxed) & unsigned(s));
}

void disable(char const * why)
{
    if (disabled.exchange(true))
        return;
    fmt::print(stderr,
            "# las-gpu: {}; offloading is disabled for the rest of this"
            " run, and the sieving carries on on the CPU\n", why);
}

gpu_engine * engine()
{
    return the_engine.get();
}

std::string describe()
{
    return the_engine ? the_engine->name() : "none";
}

} // namespace las_gpu
