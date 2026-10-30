#ifndef CADO_LAS_PROCESS_BUCKET_REGION_HPP
#define CADO_LAS_PROCESS_BUCKET_REGION_HPP

#include <functional>
#include <memory>

#include "las-where-am-i-proxy.hpp"
#include "las-auxiliary-data.hpp"
#include "sieve-methods.hpp"

/* Ideally this #ifdef should only be in las.ccp and not here */
#ifdef SIQS_SIEVE
    using ALGO = SIQS;
#else
    using ALGO = NFS;
#endif

class nfs_work;
class nfs_work_cofac;
class thread_pool;

/* before_batch, when given, is called from the calling thread before
 * each batch of bucket regions is queued, with the index of the first
 * one, and with the previous batch done. If it returns false, no
 * further region is processed. */
void process_many_bucket_regions(
        nfs_work & ws,
        std::shared_ptr<nfs_work_cofac> wc_p,
        std::shared_ptr<nfs_aux> aux_p,
        ALGO::special_q_data const & Q,
        thread_pool & pool,
        int first_region0_index,
        where_am_I & w,
        std::function<bool(int)> const & before_batch = {});

#endif	/* CADO_LAS_PROCESS_BUCKET_REGION_HPP */
