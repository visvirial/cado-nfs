/* The packed layout of spmv_gf2.cu (SPMV_GF2_LAYOUT=packed): the
 * matrix in one column band, with the column indices of the heavy rows
 * delta-coded.
 *
 * A stored row of more than LIGHT_MAX entries splits into chunks of at
 * most PK_CHUNK sorted columns.  A chunk is a 16-byte descriptor
 * (pk_chunk: its first column, its row, its entry count, the gap width
 * b, and the word offset of its data) and the data: one block of b
 * 32-bit words per PK_BLK entries, holding the PK_BLK gaps of b bits,
 * gap k = col_k - col_{k-1} within the chunk (gap 0 of the chunk is
 * zero: the first column is the descriptor's).  b is the bit length of
 * the largest gap of the chunk, so a heavy row with a mean gap g costs
 * about log2(g) + 3 bits per entry instead of the 32 of a raw index: the
 * weight of a column of a filtered NFS matrix follows a power law, and
 * about half the entries of the transposed matrix sit in rows of gaps
 * below 2^16.
 *
 * One warp gathers a chunk: in each step the lanes decode PK_D blocks
 * (lane k extracts gap k of a block from two words, an inclusive warp
 * scan gives the columns, and the running column carries from block to
 * block), so each lane keeps PK_D source loads in flight, and at the end
 * the lanes fold their sums and one atomicXor per word adds the chunk
 * into its row.
 *
 * A row of at most LIGHT_MAX entries is a light row: its columns stay
 * raw in a pool (pk_light: the row, the count, the word offset), and
 * one thread gathers the whole row, with LT_D loads in flight, and
 * stores it with no atomic.  Such rows hold the random part of the
 * matrix: with the power-law weights, most rows are light but they hold
 * under a fifth of the entries.
 *
 * The scatter (the transposed product) decodes the same structures and
 * xors the row's element into the columns.
 */

#ifndef PK_CHUNK
#define PK_CHUNK 1024
#endif
#define PK_BLK 32
#define PK_D 4
#define LT_D 8
#ifndef LIGHT_MAX
#define LIGHT_MAX 63
#endif

struct __align__(16) pk_chunk {
    uint64_t a;         /* word offset (40 bits) | (entries - 1) << 40 | b << 50 */
    uint32_t base;      /* the first column */
    uint32_t row;       /* the device row */
    __host__ __device__ uint64_t woff() const { return a & ((1ull << 40) - 1); }
    __host__ __device__ unsigned int nent() const { return (unsigned int) (a >> 40 & 1023) + 1; }
    __host__ __device__ unsigned int bits() const { return (unsigned int) (a >> 50 & 63); }
};

struct __align__(16) pk_light {
    uint64_t off;       /* word offset of the columns in the light pool */
    uint32_t row;
    uint32_t count;
};

static inline uint64_t pk_pack(uint64_t woff, unsigned int nent, unsigned int b)
{
    return woff | (uint64_t) (nent - 1) << 40 | (uint64_t) b << 50;
}

/* the gap of lane k of a block of b words at blk, b < 32 */
__device__ __forceinline__ uint32_t pk_gap(const uint32_t * blk, unsigned int b,
        unsigned int k)
{
    if (b == 0)
        return 0;
    unsigned int const bit = k * b;
    unsigned int const q = bit >> 5, r = bit & 31;
    uint32_t const lo = __ldg(blk + q);
    uint32_t const hi = r + b > 32 ? __ldg(blk + q + 1) : 0;
    return __funnelshift_r(lo, hi, r) & ((1u << b) - 1);
}

/* inclusive warp scan of x */
__device__ __forceinline__ uint32_t pk_scan(uint32_t x)
{
    unsigned const lane = threadIdx.x & 31;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        uint32_t const y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    return x;
}

/* A lane loads A = W / P words of an element: at W = 4 a lane pair
 * shares each element, so the two 16-byte loads of one 32-byte element
 * are one request of one instruction (one L1 sector access), as in
 * gather_kernel; one thread issuing both loads costs two requests. */
template <int W, int P>
__device__ __forceinline__ void pk_load(const uint64_t * src, uint32_t col,
        unsigned int part, uint64_t (&v)[W / P])
{
    constexpr int A = W / P;
    if constexpr (A % 2 == 0) {
#pragma unroll
        for (int h = 0; h < A / 2; h++) {
            ulonglong2 const x = ((const ulonglong2 *) src)
                [(size_t) col * (W / 2) + part * (A / 2) + h];
            v[2 * h] = x.x;
            v[2 * h + 1] = x.y;
        }
    } else {
        v[0] = src[(size_t) col * W + part];
    }
}

/* One warp per chunk (blockDim.x a multiple of 32).  Each step decodes
 * PK_D blocks: the gap loads of all blocks go out first, then the scans
 * chain through the carry; then the lanes gather the 32 * PK_D columns,
 * P lanes per element (P = 2 at W = 4), 32 / P elements per pass, and
 * each lane keeps PK_D * P loads in flight. */
template <int W>
__global__ void __launch_bounds__(128)
pk_gather_kernel(const pk_chunk * chunks, const uint32_t * pool,
        const uint64_t * src, uint64_t * dst, unsigned int nchunks)
{
    constexpr int P = W == 4 ? 2 : 1;
    constexpr int A = W / P;
    unsigned int const c = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned int const lane = threadIdx.x & 31;
    unsigned int const part = lane % P;
    if (c >= nchunks) return;
    pk_chunk const g = chunks[c];
    unsigned int const nent = g.nent(), b = g.bits();
    const uint32_t * blk = pool + g.woff();
    uint32_t carry = g.base;
    uint64_t acc[A] = {};
    for (unsigned int e0 = 0; e0 < nent; e0 += PK_D * PK_BLK) {
        uint32_t gap[PK_D];
#pragma unroll
        for (int u = 0; u < PK_D; u++)
            gap[u] = e0 + u * PK_BLK < nent
                ? pk_gap(blk + (size_t) (e0 / PK_BLK + u) * b, b, lane) : 0;
        uint32_t col[PK_D];
#pragma unroll
        for (int u = 0; u < PK_D; u++) {
            col[u] = carry + pk_scan(gap[u]);
            carry = __shfl_sync(0xffffffffu, col[u], 31);
        }
        uint64_t v[PK_D][P][A] = {};
#pragma unroll
        for (int u = 0; u < PK_D; u++) {
#pragma unroll
            for (int h = 0; h < P; h++) {
                unsigned int const e = h * (32 / P) + lane / P;  /* entry of this pass */
                uint32_t const cc = __shfl_sync(0xffffffffu, col[u], e);
                if (e0 + u * PK_BLK + e < nent)
                    pk_load<W, P>(src, cc, part, v[u][h]);
            }
        }
#pragma unroll
        for (int u = 0; u < PK_D; u++)
#pragma unroll
            for (int h = 0; h < P; h++)
#pragma unroll
                for (int w = 0; w < A; w++)
                    acc[w] ^= v[u][h][w];
    }
    /* fold the lanes of the same part; lanes 0 .. P-1 add their words */
#pragma unroll
    for (int w = 0; w < A; w++)
#pragma unroll
        for (int o = 16; o >= P; o >>= 1)
            acc[w] ^= __shfl_xor_sync(0xffffffffu, acc[w], o);
    if (lane < P)
#pragma unroll
        for (int w = 0; w < A; w++)
            atomicXor((unsigned long long *) &dst[(size_t) g.row * W + part * A + w],
                    (unsigned long long) acc[w]);
}

/* P threads per light row (a lane pair at W = 4), LT_D elements in
 * flight each; the row goes to dst with plain stores */
template <int W>
__global__ void lt_gather_kernel(const pk_light * rows, const uint32_t * pool,
        const uint64_t * src, uint64_t * dst, unsigned int nrows)
{
    constexpr int P = W == 4 ? 2 : 1;
    constexpr int A = W / P;
    unsigned int const t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int const i = t / P, part = t % P;
    if (i >= nrows) return;
    pk_light const g = rows[i];
    const uint32_t * cols = pool + g.off;
    uint64_t acc[A] = {};
    for (unsigned int e0 = 0; e0 < g.count; e0 += LT_D) {
        uint32_t c[LT_D];
#pragma unroll
        for (int u = 0; u < LT_D; u++)
            c[u] = e0 + u < g.count ? __ldcs(cols + e0 + u) : 0;
        uint64_t v[LT_D][A] = {};
#pragma unroll
        for (int u = 0; u < LT_D; u++)
            if (e0 + u < g.count)
                pk_load<W, P>(src, c[u], part, v[u]);
#pragma unroll
        for (int u = 0; u < LT_D; u++)
#pragma unroll
            for (int w = 0; w < A; w++)
                acc[w] ^= v[u][w];
    }
    if constexpr (A % 2 == 0) {
#pragma unroll
        for (int h = 0; h < A / 2; h++)
            ((ulonglong2 *) dst)[(size_t) g.row * (W / 2) + part * (A / 2) + h] =
                make_ulonglong2(acc[2 * h], acc[2 * h + 1]);
    } else {
        dst[(size_t) g.row * W + part] = acc[0];
    }
}

/* the transposed product: dst[col] ^= src[row] over the entries */
template <int W>
__global__ void pk_scatter_kernel(const pk_chunk * chunks, const uint32_t * pool,
        const uint64_t * src, uint64_t * dst, unsigned int nchunks)
{
    unsigned int const c = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned int const lane = threadIdx.x & 31;
    if (c >= nchunks) return;
    pk_chunk const g = chunks[c];
    unsigned int const nent = g.nent(), b = g.bits();
    const uint32_t * blk = pool + g.woff();
    uint64_t val[W];
    for (int w = 0; w < W; w++) val[w] = src[(size_t) g.row * W + w];
    uint32_t carry = g.base;
    for (unsigned int e0 = 0; e0 < nent; e0 += PK_BLK) {
        uint32_t const gap = pk_gap(blk + (size_t) (e0 / PK_BLK) * b, b, lane);
        uint32_t const col = carry + pk_scan(gap);
        carry = __shfl_sync(0xffffffffu, col, 31);
        if (e0 + lane < nent)
            for (int w = 0; w < W; w++)
                atomicXor((unsigned long long *) &dst[(size_t) col * W + w],
                        (unsigned long long) val[w]);
    }
}

template <int W>
__global__ void lt_scatter_kernel(const pk_light * rows, const uint32_t * pool,
        const uint64_t * src, uint64_t * dst, unsigned int nrows)
{
    unsigned int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrows) return;
    pk_light const g = rows[i];
    const uint32_t * cols = pool + g.off;
    uint64_t val[W];
    for (int w = 0; w < W; w++) val[w] = src[(size_t) g.row * W + w];
    for (unsigned int e = 0; e < g.count; e++) {
        uint32_t const col = cols[e];
        for (int w = 0; w < W; w++)
            atomicXor((unsigned long long *) &dst[(size_t) col * W + w],
                    (unsigned long long) val[w]);
    }
}

/* The packed data of one device: built on the host from the sorted row
 * stream, then uploaded. */
struct pk_build {
    std::vector<pk_chunk> chunks;
    std::vector<uint32_t> pool;         /* the gap words */
    std::vector<pk_light> lights;
    std::vector<uint32_t> lpool;        /* the raw columns of the light rows */
    std::vector<uint32_t> chunk_row;    /* the device row of each chunk */
    std::vector<uint32_t> light_row;    /* the device row of each light row */
};

static inline unsigned int pk_bitlen(uint32_t x)
{
    return x ? 32 - __builtin_clz(x) : 0;
}

/* the per-chunk width and size of row i: fills chunks, returns the data
 * words */
static uint64_t pk_row_chunks(const uint32_t * cols, uint64_t len,
        unsigned int row, uint64_t woff, std::vector<pk_chunk> * out)
{
    uint64_t words = 0;
    for (uint64_t s = 0; s < len; s += PK_CHUNK) {
        unsigned int const n = (unsigned int) std::min<uint64_t>(PK_CHUNK, len - s);
        uint32_t maxgap = 0;
        for (unsigned int k = 1; k < n; k++)
            maxgap = std::max(maxgap, cols[s + k] - cols[s + k - 1]);
        unsigned int const b = pk_bitlen(maxgap);
        if (b > 31) {
            fprintf(stderr, "spmv_gf2: a column gap of 32 bits\n");
            abort();
        }
        uint64_t const nblk = (n + PK_BLK - 1) / PK_BLK;
        if (out)
            out->push_back({ pk_pack(woff + words, n, b), cols[s], row });
        words += nblk * b;
    }
    return words;
}

/* write the gap words of row i's chunks at p */
static void pk_row_fill(const uint32_t * cols, uint64_t len, uint32_t * p)
{
    for (uint64_t s = 0; s < len; s += PK_CHUNK) {
        unsigned int const n = (unsigned int) std::min<uint64_t>(PK_CHUNK, len - s);
        uint32_t maxgap = 0;
        for (unsigned int k = 1; k < n; k++)
            maxgap = std::max(maxgap, cols[s + k] - cols[s + k - 1]);
        unsigned int const b = pk_bitlen(maxgap);
        uint64_t const nblk = (n + PK_BLK - 1) / PK_BLK;
        std::fill(p, p + nblk * b, 0);
        for (unsigned int k = 1; k < n && b; k++) {
            uint32_t const gap = cols[s + k] - cols[s + k - 1];
            unsigned int const j = k / PK_BLK, l = k % PK_BLK;
            uint64_t const bit = (uint64_t) l * b;
            uint32_t * const w = p + j * b + (bit >> 5);
            unsigned int const r = bit & 31;
            w[0] |= gap << r;
            if (r + b > 32)
                w[1] |= gap >> (32 - r);
        }
        p += nblk * b;
    }
}

/* Build the packed layout of rows [row0, row1) of the stream on nthr
 * threads: a count pass gives each thread its offsets, a fill pass
 * writes.  The stream rows are sorted. */
static pk_build pk_build_rows(row_stream const & rs, unsigned int row0,
        unsigned int row1, unsigned int nthr)
{
    pk_build B;
    std::vector<unsigned int> cut(nthr + 1, row1);
    {
        uint64_t const w0 = rs.rowptr[row0], w1 = rs.rowptr[row1];
        cut[0] = row0;
        for (unsigned int t = 1; t < nthr; t++)
            cut[t] = std::upper_bound(rs.rowptr.begin() + cut[t - 1],
                    rs.rowptr.begin() + row1, w0 + (w1 - w0) / nthr * t)
                - rs.rowptr.begin();
    }
    struct counts { uint64_t nchunk = 0, words = 0, nlight = 0, lwords = 0; };
    std::vector<counts> cnt(nthr);
    for_threads(nthr, [&](unsigned int t) {
        counts & c = cnt[t];
        for (unsigned int i = cut[t]; i < cut[t + 1]; i++) {
            uint64_t const len = rs.len(i);
            if (!len)
                continue;
            if (len <= LIGHT_MAX) {
                c.nlight++;
                c.lwords += len;
            } else {
                c.nchunk += (len + PK_CHUNK - 1) / PK_CHUNK;
                c.words += pk_row_chunks(rs.cols(i), len, i, 0, nullptr);
            }
        }
    });
    std::vector<counts> off(nthr + 1);
    for (unsigned int t = 0; t < nthr; t++) {
        off[t + 1].nchunk = off[t].nchunk + cnt[t].nchunk;
        off[t + 1].words = off[t].words + cnt[t].words;
        off[t + 1].nlight = off[t].nlight + cnt[t].nlight;
        off[t + 1].lwords = off[t].lwords + cnt[t].lwords;
    }
    if (off[nthr].words >= 1ull << 40) {
        fprintf(stderr, "spmv_gf2: the packed data passes 2^40 words\n");
        abort();
    }
    B.chunks.resize(off[nthr].nchunk);
    B.chunk_row.resize(off[nthr].nchunk);
    B.pool.resize(off[nthr].words);
    B.lights.resize(off[nthr].nlight);
    B.light_row.resize(off[nthr].nlight);
    B.lpool.resize(off[nthr].lwords);
    for_threads(nthr, [&](unsigned int t) {
        std::vector<pk_chunk> ch;
        uint64_t woff = off[t].words, lwoff = off[t].lwords;
        uint64_t nl = off[t].nlight, nc = off[t].nchunk;
        for (unsigned int i = cut[t]; i < cut[t + 1]; i++) {
            uint64_t const len = rs.len(i);
            if (!len)
                continue;
            if (len <= LIGHT_MAX) {
                B.lights[nl] = { lwoff, i - row0, (uint32_t) len };
                B.light_row[nl++] = i - row0;
                std::copy(rs.cols(i), rs.cols(i) + len, B.lpool.data() + lwoff);
                lwoff += len;
            } else {
                ch.clear();
                uint64_t const words = pk_row_chunks(rs.cols(i), len, i - row0,
                        woff, &ch);
                pk_row_fill(rs.cols(i), len, B.pool.data() + woff);
                for (pk_chunk const & c : ch) {
                    B.chunk_row[nc] = i - row0;
                    B.chunks[nc++] = c;
                }
                woff += words;
            }
        }
    });
    return B;
}
