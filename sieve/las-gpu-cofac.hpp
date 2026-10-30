#ifndef CADO_LAS_GPU_COFAC_HPP
#define CADO_LAS_GPU_COFAC_HPP

#include <memory>
#include <mutex>
#include <vector>

#include "las-cofac-standalone.hpp"

class nfs_aux;
class nfs_work_cofac;
class thread_pool;

/* Cofactorization with the device doing the ECM curves it can do.
 *
 * Nearly all the time of the cofactorization goes to the MONTY12 curves
 * of the default strategy, on numbers that a device handles at a couple
 * of hundred times the speed of one CPU thread -- provided it gets them
 * by the tens of thousands, and one curve at a time. So instead of
 * having one task per survivor as detached_cofac does, the survivors of
 * a special-q are held back here until its bucket regions are done, and
 * then taken through their strategies together: whichever of them need
 * the same curve next go to the device in one batch. The rest of what
 * they need (the other methods, what is done with a factor found, the
 * numbers that split) stays on the CPU, in ordinary tasks of the ECM
 * queue.
 *
 * The factors are the ones that the CPU would have found (see
 * sieve/gpu/las-gpu-ecm.hpp), so that the relations are the same.
 *
 * nfs_work_cofac holds one of these per attempt at a special-q, so that
 * the survivors of a failed attempt are just dropped: the next attempt
 * finds them again.
 */
class las_gpu_cofac {
    public:
    /* nullptr when the device is not to be used for this right now */
    static std::shared_ptr<las_gpu_cofac> create();

    /* Thread-safe. */
    void add(cofac_standalone && cur);

    /* Sets the survivors added so far on their way. Returns at once. */
    void flush(std::shared_ptr<nfs_work_cofac> const & wc_p,
               std::shared_ptr<nfs_aux> const & aux_p,
               thread_pool & pool);

    private:
    std::mutex mu;
    std::vector<cofac_standalone> survivors;
};

#endif /* CADO_LAS_GPU_COFAC_HPP */
