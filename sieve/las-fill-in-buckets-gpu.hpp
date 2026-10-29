#ifndef CADO_LAS_FILL_IN_BUCKETS_GPU_HPP
#define CADO_LAS_FILL_IN_BUCKETS_GPU_HPP

#include <memory>

#include "las-qlattice.hpp"
#include "las-where-am-i-proxy.hpp"

class nfs_aux;
class nfs_work;
class worker_thread;

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

/* Sieve every slice that las_gpu_fib_takes_slice() accepts, into one
 * bucket array. Throws buckets_are_full exactly as the ordinary path
 * does, and with the same information, so that the caller's retry loop
 * does not need to know where the updates came from.
 */
void las_gpu_fib_run(worker_thread * worker, int side, nfs_work & ws,
                     nfs_aux & aux, qlattice_basis const & Q,
                     std::shared_ptr<where_am_I> w);

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
