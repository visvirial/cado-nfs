#ifndef CADO_LAS_FILL_IN_BUCKETS_GPU_HPP
#define CADO_LAS_FILL_IN_BUCKETS_GPU_HPP

#include <memory>

#include "las-qlattice.hpp"
#include "las-where-am-i-proxy.hpp"

class nfs_aux;
struct las_info;
struct siever_config;
class nfs_work;
class thread_pool;

/* Toplevel bucket sieving on the backend that -gpu selected.
 *
 * fill_in_buckets_toplevel() is where las spends about half of its
 * time, and it is the one stage whose shape suits a device: the work is
 * one independent Franke-Kleinjung walk per (factor base entry, root)
 * pair, and there are tens of thousands of those. Better still, the
 * factor base does not depend on the special-q, so the device holds it
 * from one q to the next and a pass costs one small upload (the
 * q-lattice) and one large download (the updates).
 *
 * What the device does not do is handed back: slices of general factor
 * base entries, which are a different and much rarer shape, and
 * projective-like lattices, which produce row updates rather than
 * ordinary ones. Both go through the unmodified CPU code.
 */

/* The slice types the device takes. The caller keeps pushing ordinary
 * per-slice tasks for the others, so that the two paths together cover
 * the part exactly once.
 */
template <typename ENTRY_TYPE>
constexpr bool las_gpu_fib_takes_slice()
{
    return !ENTRY_TYPE::is_general_type && ENTRY_TYPE::fixed_nr_roots >= 1;
}

/* Whether the device path applies at all: a backend was selected, the
 * toplevel is 1, there are no sublattices, and we are not tracing.
 */
bool las_gpu_fib_usable(nfs_work const & ws, qlattice_basis const & Q);

/* The weight -- expected updates per sieve cell, the unit of
 * fb_factorbase::slicing::stats.weight -- of the slices that the device
 * will take on this side, or 0 if it takes none. The level-1 bucket
 * arrays need no room for those: the device's updates stay in its own
 * buffer. */
double las_gpu_fib_weight(nfs_work const & ws, int side,
                          qlattice_basis const & Q);

/* The same, for a special-q that is not being sieved yet, as far as it
 * can be told before prepare_for_new_q(). */
bool las_gpu_fib_usable_for(siever_config const & conf,
                            qlattice_basis const & Q);

/* Launch the device pass over every slice that
 * las_gpu_fib_takes_slice() accepts, and queue the task that will hand
 * its output to the bucket arrays once it is back.
 *
 * The launch does not wait for the device, so whatever the caller
 * queues next -- the ordinary tasks for the other slices, the other
 * side -- runs while the device works. The collecting task has cost 0,
 * so the pool takes it last.
 *
 * The bucket arrays do not get a copy of the updates: they point into
 * the pass's own buffer, which nfs_work::side_data keeps until the next
 * special-q. The device's updates therefore take no room in the arrays
 * and can never make them overflow.
 *
 * Returns the total weight of the slices the device took, 0 if none.
 */
double las_gpu_fib_start(nfs_work & ws, nfs_aux & aux,
                         qlattice_basis const & Q, int side,
                         thread_pool & pool);

/* Launch, ahead of time, the pass for a special-q whose sieve area has
 * been chosen but which is not being sieved yet -- typically the next
 * one, while the current one's bucket regions are being processed. The
 * device's share of bucket sieving then costs nothing on the critical
 * path, which is what keeps it from being the bottleneck when every
 * core is busy.
 *
 * Nothing about the current special-q is touched. The pass is kept in
 * nfs_work::side_data, and las_gpu_fib_start() adopts it if it turns
 * out to be the one it needs: same slicing, same plan. Otherwise it is
 * ignored, and replaced by the next prefetch.
 */
void las_gpu_fib_prefetch(las_info & las, nfs_work & ws,
                          siever_config const & conf,
                          qlattice_basis const & Q, uint32_t J);

/* Device-side state that outlives a special-q, hung off
 * nfs_work::side_data. Opaque here so that the sieve headers stay free
 * of the gpu/ ones.
 */
struct las_gpu_fib_cache;
struct las_gpu_fib_cache_deleter {
    void operator()(las_gpu_fib_cache *) const;
};
using las_gpu_fib_cache_ptr =
    std::unique_ptr<las_gpu_fib_cache, las_gpu_fib_cache_deleter>;

#endif /* CADO_LAS_FILL_IN_BUCKETS_GPU_HPP */
