#include "cado.h" // IWYU pragma: keep

#include <cmath>
#include <cstdint>

#include "cado_poly.hpp"
#include "las-config.hpp"
#include "las-norms-gpu.hpp"
#include "las-qlattice.hpp"
#include "las-siever-config.hpp"
#include "polynomial.hpp"
#include "macros.h"
#include "gpu/las-gpu-plan.hpp"
#include "gpu/las-gpu-select.hpp"

static_assert(CADO_GPU_LOGNORM_GUARD_BITS == LOGNORM_GUARD_BITS,
              "the device-side copy of LOGNORM_GUARD_BITS has drifted");

lognorm_gpu::lognorm_gpu(siever_config const & sc, cxx_cado_poly const & cpoly,
                         int side, special_q_data_class auto const & Q,
                         int logI, uint32_t J)
  : lognorm_smart(sc, cpoly, side, Q, logI, J)
{
    build_plan();

    eng = las_gpu::engine();
    if (eng)
        task = eng->norm_task_create(plan, segments.data(),
                                     int(segments.size()));
}

template lognorm_gpu::lognorm_gpu(siever_config const &, cxx_cado_poly const &,
                                  int, qlattice_basis const &, int, uint32_t);
template lognorm_gpu::lognorm_gpu(siever_config const &, cxx_cado_poly const &,
                                  int, siqs_special_q_data const &, int,
                                  uint32_t);

lognorm_gpu::~lognorm_gpu()
{
    if (eng && task)
        eng->norm_task_destroy(task);
}

void lognorm_gpu::build_plan()
{
    plan.logI = logI;
    plan.log_bucket_region = LOG_BUCKET_REGION;
    plan.degree = fijd.degree();
    plan.scale = scale;
    plan.modscale = scale / 0x100000;
    plan.offset = 0x3FF00000 - LOGNORM_GUARD_BITS / plan.modscale;

    /* Same dispatch as lognorm_smart::fill(). */
    if (J == 1)
        plan.kind = gpu_norm_plan::KIND_SIQS;
    else if (fijd.degree() > 1)
        plan.kind = gpu_norm_plan::KIND_ALG;
    else
        plan.kind = gpu_norm_plan::KIND_RAT;

    /* The single meaningful cell of the line j = 0, see
     * LOGNORM_COMMON_HANDLE_ORIGIN in las-norms.cpp. */
    plan.origin_value =
        LOGNORM_GUARD_BITS +
        (unsigned char) (std::log2(std::fabs(fijd.lc())) * scale);

    /* fijd[i] hands out a coeff_proxy, which converts to double only
     * when we ask it to; letting it take part in a conditional
     * expression picks a conversion we do not want. Go through the
     * const accessor and cast explicitly. */
    polynomial<double> const & F(fijd);
    plan.u0 = F.degree() >= 0 ? double(F[0]) : 0.;
    plan.u1 = F.degree() >= 1 ? double(F[1]) : 0.;

    segments.clear();
    if (plan.kind != gpu_norm_plan::KIND_RAT) {
        ASSERT_ALWAYS(G.endpoints.size() == G.equations.size() + 1);
        segments.reserve(G.equations.size());
        for (size_t k = 0; k < G.equations.size(); k++)
            segments.push_back({ G.endpoints[k], G.endpoints[k + 1],
                                 G.equations[k].first,
                                 G.equations[k].second });
    }
    plan.segments = segments.data();
    plan.nsegments = int(segments.size());
}

void lognorm_gpu::fill(unsigned char * S, unsigned int N,
                       sublat_runtime_t const & sl) const
{
    if (sl.m != 1) {
        lognorm_smart::fill(S, N, sl);
        return;
    }
    fill_many(S, N, 1);
}

void lognorm_gpu::fill_many(unsigned char * S, unsigned int N0,
                            unsigned int nregions) const
{
    if (!task) {
        /* No backend selected: behave exactly like our base class. */
        for (unsigned int r = 0; r < nregions; r++)
            lognorm_smart::fill(S + (size_t(r) << LOG_BUCKET_REGION), N0 + r,
                                sublat_runtime_t::no_sublat());
        return;
    }
    eng->norm_fill(task, S, N0, nregions);
}
