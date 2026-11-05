/* Correctness test of the spmv_gf2 device backend against a plain host
 * product: a random row stream (uniform rows, or the power-law row
 * weights of SPMV_GF2_RANDOM_ALPHA through random_matrix_stream_solib),
 * the gather and the scatter at words 1, 2 and 4, compared bit for bit.
 * The environment selects the layout and kernel variants under test,
 * as in bwc (SPMV_GF2_BAND_MB, SPMV_GF2_CHUNK, ...).
 *
 *   spmv_test [nrows [density [range [seed]]]]
 *
 * exits 0 when every product matches. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>

#include "spmv_gf2.h"

/* the arena registry of bwc is absent here: the backend takes the host
 * path (bwc_device_arena.hpp) */
extern "C" int bwc_dev_allowed() { return 0; }
extern "C" struct bwc_device_arena * bwc_dev_find(const void *) { return nullptr; }
extern "C" struct bwc_device_arena * bwc_dev_register(struct bwc_device_arena const &)
{ abort(); }

/* The backend page-locks the host vectors it sees and keeps the lock
 * for the process (bwc keeps its vectors for the run), so the vectors
 * live for the whole program: a freed locked range would break a later
 * copy into memory reallocated there. */
static std::vector<uint64_t> x, y, ref;

static int run(unsigned int nrows, double density, unsigned int range,
        unsigned long seed, unsigned int words)
{
    std::vector<uint32_t> stream = random_matrix_stream_solib(nrows, range,
            density, seed);
    uint64_t const nnz = stream.size() - nrows;
    unsigned int const max_dim = nrows > range ? nrows : range;
    size_t const w = words;

    std::mt19937_64 rng(seed * 7919 + words);
    for (size_t i = 0; i < (size_t) max_dim * w; i++) x[i] = rng();

    /* reference gather: y[i] = xor of x[col] over row i */
    {
        size_t k = 0;
        for (unsigned int i = 0; i < nrows; i++) {
            uint32_t const len = stream[k++];
            for (size_t j = 0; j < w; j++) ref[i * w + j] = 0;
            for (uint32_t t = 0; t < len; t++, k++)
                for (size_t j = 0; j < w; j++)
                    ref[i * w + j] ^= x[(size_t) stream[k] * w + j];
        }
        for (size_t i = (size_t) nrows * w; i < (size_t) max_dim * w; i++) ref[i] = 0;
    }
    /* the setup may reorder the columns of a row; the reference above
     * ran before it */
    spmv_gf2 * m = spmv_gf2_new(nrows, stream.size(), stream.data(), max_dim,
            words);
    int bad = 0;
    for (int rep = 0; rep < 3; rep++) {
        std::fill(y.begin(), y.begin() + (size_t) max_dim * w, 0xdeadbeefull);
        spmv_gf2_gather(m, y.data(), x.data(), max_dim, range);
        size_t diff = 0;
        for (size_t i = 0; i < (size_t) max_dim * w; i++)
            diff += y[i] != ref[i];
        printf("words %u gather %d: %s (%zu of %zu words differ)\n", words,
                rep, diff ? "FAIL" : "ok", diff, (size_t) max_dim * w);
        bad += diff != 0;
    }
    /* reference scatter: z[col] ^= x[i] over the entries col of row i */
    {
        std::fill(ref.begin(), ref.begin() + (size_t) max_dim * w, 0);
        /* the stream may be reordered now; the reference only needs the
         * multiset of (row, col) */
        size_t k = 0;
        for (unsigned int i = 0; i < nrows; i++) {
            uint32_t const len = stream[k++];
            for (uint32_t t = 0; t < len; t++, k++)
                for (size_t j = 0; j < w; j++)
                    ref[(size_t) stream[k] * w + j] ^= x[i * w + j];
        }
        std::fill(y.begin(), y.begin() + (size_t) max_dim * w, 0xdeadbeefull);
        spmv_gf2_scatter(m, y.data(), x.data(), range, nrows);
        size_t diff = 0;
        for (size_t i = 0; i < (size_t) range * w; i++)
            diff += y[i] != ref[i];
        printf("words %u scatter: %s (%zu of %zu words differ)\n", words,
                diff ? "FAIL" : "ok", diff, (size_t) range * w);
        bad += diff != 0;
    }
    spmv_gf2_free(m);
    printf("words %u: %lu rows, %lu entries, %s\n", words,
            (unsigned long) nrows, (unsigned long) nnz, bad ? "FAIL" : "ok");
    return bad;
}

int main(int argc, char ** argv)
{
    unsigned int const nrows = argc > 1 ? atoi(argv[1]) : 200000;
    double const density = argc > 2 ? atof(argv[2]) : 40;
    unsigned int const range = argc > 3 ? atoi(argv[3]) : nrows;
    unsigned long const seed = argc > 4 ? atol(argv[4]) : 1;
    size_t const max_dim = nrows > range ? nrows : range;
    x.resize(max_dim * 4);
    y.resize(max_dim * 4);
    ref.resize(max_dim * 4);
    int bad = 0;
    for (unsigned int words : { 1u, 2u, 4u })
        bad += run(nrows, density, range, seed, words);
    printf("%s\n", bad ? "SOME PRODUCTS FAILED" : "all products match");
    return bad != 0;
}
