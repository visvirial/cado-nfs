#include "cado.h" // IWYU pragma: keep

#include <memory>
#include <stdexcept>
#include <string>

#include "fmt/format.h"

#include "las-gpu-select.hpp"

namespace las_gpu {

namespace {
std::unique_ptr<gpu_engine> the_engine;
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
        the_engine.reset(gpu_engine_cpu_create());
    }
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
