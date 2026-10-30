#ifndef CADO_LAS_DETACHED_COFAC_HPP
#define CADO_LAS_DETACHED_COFAC_HPP

#include <memory>
#include <vector>

#include "las-cofac-standalone.hpp"

class nfs_aux;
class nfs_work_cofac;
class worker_thread;
struct relation;

/* What detached_cofac does once it knows what became of the leftover
 * norms: pass is what cofac_standalone::factor_leftover_norms returned,
 * and cof_bitsize what las.cofac_stats.call() filled in. */
extern relation detached_cofac_conclude(worker_thread * worker,
        nfs_work_cofac & wc,
        nfs_aux & aux,
        cofac_standalone & cur,
        int pass,
        std::vector<int> const & cof_bitsize);

extern relation detached_cofac(worker_thread * worker,
        nfs_work_cofac & wc,
        nfs_aux & aux,
        cofac_standalone && cur);

#endif	/* CADO_LAS_DETACHED_COFAC_HPP */
