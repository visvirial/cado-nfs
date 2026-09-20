#ifndef CADO_LAS_GPU_ARCH_HPP
#define CADO_LAS_GPU_ARCH_HPP

/* Portability shims that let the very same source code be compiled both
 * as ordinary host C++ and as CUDA device code.
 *
 * The files in this directory are deliberately kept free of any cado-nfs
 * header (only <cstdint> and friends are allowed), because they are fed
 * to nvcc, which we do not want to expose to the C++20 machinery that
 * the rest of the sieve code relies on.
 */

#if defined(__CUDACC__)
#define CADO_GPU_FN __host__ __device__ inline
#define CADO_GPU_HOST_FN __host__ inline
#else
#define CADO_GPU_FN inline
#define CADO_GPU_HOST_FN inline
#endif

/* The whole point of the two backends is that they must agree bit for
 * bit. The one thing that can silently break that is the contraction of
 * a*b+c into an fma, which nvcc does by default and which host compilers
 * may do as well. Both backends are compiled with contraction disabled
 * (-fmad=false / -ffp-contract=off, see sieve/CMakeLists.txt), and the
 * pragma below is a second line of defence for the host side.
 */
#if defined(__clang__)
#define CADO_GPU_NO_FP_CONTRACT _Pragma("clang fp contract(off)")
#else
#define CADO_GPU_NO_FP_CONTRACT
#endif

#endif /* CADO_LAS_GPU_ARCH_HPP */
