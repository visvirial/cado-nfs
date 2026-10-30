# Optional CUDA support for the offloaded sieve stages (see sieve/gpu).
#
# We do not probe for nvcc unless we are asked to: the vast majority of
# builds do not want it, and check_language() is not free. Enable with
#
#       ENABLE_CUDA=1 make
#
# or -DENABLE_CUDA=1 on the cmake command line. A las built this way
# offloads bucket sieving to the first usable CUDA device by default
# (-gpu auto), and sieves on the CPU alone when there is none, or when
# the device runs out of memory; pass -gpu none to keep it off the
# device altogether. CUDA_ARCH, if set, is
# passed on as CMAKE_CUDA_ARCHITECTURES ("native" by default, which
# requires a visible device at configure time; set it explicitly, e.g.
# CUDA_ARCH=80, when cross-building for a machine you do not have).
#
# check_language() picks the first nvcc on PATH, which on a distribution
# that also ships a cuda toolkit package is quite likely to be an older
# one than the toolkit under /usr/local/cuda. That matters because nvcc
# rejects host compilers newer than the ones its release knew about, so
# an old nvcc next to a recent gcc/clang fails outright. Point CUDACXX
# at the toolkit you mean when the two disagree, e.g.
#
#       ENABLE_CUDA=1 CUDACXX=/usr/local/cuda/bin/nvcc make

set(HAVE_CUDA 0)

if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        if(CUDA_ARCH)
            set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
        else()
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        # Bit-for-bit agreement with the host backend is the whole point
        # of sharing las-gpu-kernels.hpp between the two, and an fma
        # contraction would silently break it.
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fmad=false")
        # Anchor the toolkit lookup to the nvcc we actually selected.
        # Without this, find_package() happily pairs that nvcc with the
        # libcudart.so of a *different* toolkit, because a distribution
        # cuda package puts one in the default linker path and that is
        # searched before the toolkit's own lib64.
        get_filename_component(_cuda_bindir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
        get_filename_component(CUDAToolkit_ROOT "${_cuda_bindir}" DIRECTORY)
        unset(_cuda_bindir)
        find_package(CUDAToolkit REQUIRED)
        set(HAVE_CUDA 1)
        message(STATUS "CUDA: ${CMAKE_CUDA_COMPILER}"
                       " (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
    else()
        message(WARNING "ENABLE_CUDA is set but no CUDA compiler was found;"
                        " the sieve will only have the cpu gpu-backend")
    endif()
endif()
