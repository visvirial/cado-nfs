#ifndef CADO_BUCKET_PUSH_UPDATE_HPP
#define CADO_BUCKET_PUSH_UPDATE_HPP

#include "bucket.hpp"

#ifdef SAFE_BUCKET_ARRAYS
#include <cstdio>
#endif

#include "las-where-am-i.hpp"
#include "macros.h"

template <int LEVEL, hint_type HINT>
inline void
bucket_array_t<LEVEL, HINT>::push_update(int const i, update_t const & update,
                                         where_am_I & w MAYBE_UNUSED)
{
#ifdef SAFE_BUCKET_ARRAYS
    if (bucket_write[i] >= bucket_start[i + 1]) {
        fprintf(stderr, "# Warning: hit end of bucket nb %d\n", i);
        ASSERT_ALWAYS(0);
        return;
    }
#endif
#ifdef TRACE_K
    log_this_update(update, i, w);
#endif
    if constexpr (is_delta_coded) {
        /* unsigned, so that this would remain correct, albeit costly, for
         * hints that do not come in increasing order */
        slice_offset_t delta = update.hint - last_hint[i];
        last_hint[i] = update.hint;
        if (UNLIKELY(delta >= delta_coded_update_t::escape)) {
            *bucket_write[i]++ =
                delta_coded_update_t(delta, delta_coded_update_t::escape);
            delta = 0;
        }
        *bucket_write[i]++ = delta_coded_update_t(update.x, delta);
    } else {
        *bucket_write[i]++ = update;
    }
}

template <int LEVEL, hint_type_general HINT>
inline void bucket_single<LEVEL, HINT>::push_update(update_t const & update)
{
#ifdef SAFE_BUCKETS_SINGLE
    if (start + _size <= write) {
        fprintf(stderr, "# Warning: hit end of bucket\n");
        ASSERT_ALWAYS(0);
        write--;
    }
#endif
    *(write++) = update;
}

template <int LEVEL, hint_type_general HINT>
inline typename bucket_single<LEVEL, HINT>::update_t const &
bucket_single<LEVEL, HINT>::get_next_update()
{
#ifdef SAFE_BUCKETS_SINGLE
    ASSERT_ALWAYS(read < write);
#endif
    return *read++;
}

#endif /* CADO_BUCKET_PUSH_UPDATE_HPP */
