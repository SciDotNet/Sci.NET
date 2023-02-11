find_package(OpenMP)
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/*.c
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/*.h
        ${CMAKE_CURRENT_LIST_DIR}/src/*.hpp)

find_library(CUDA_JIT_COMPILER_LIBRARY
        NAMES nvJitLink_static
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}
        ENV CUDA_PATH
        PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)

find_library(CUDA_PTX_COMPILER_LIBRARY
        NAMES nvptxcompiler_static
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}
        ENV CUDA_PATH
        PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)

find_library(CUDA_NVRTC_LIBRARY
        NAMES nvrtc_static
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}
        ENV CUDA_PATH
        PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)

find_library(CUDA_NVRTC_BUILTINS_LIBRARY
        NAMES nvrtc-builtins_static
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}
        ENV CUDA_PATH
        PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)

add_library(Sci.NET.LinearAlgebra.Native.CUDA SHARED
        ${SRC_FILES}
        ${HEADER_FILES})

target_link_libraries(Sci.NET.LinearAlgebra.Native.CUDA PUBLIC
        Sci.NET.LinearAlgebra.Native
        OpenMP::OpenMP_CXX
        CUDA::cudart_static
        ${CUDA_NVRTC_LIBRARY}
        ${CUDA_NVRTC_BUILTINS_LIBRARY}
        ${CUDA_JIT_COMPILER_LIBRARY}
        ${CUDA_PTX_COMPILER_LIBRARY})

if (WIN32)
    target_link_libraries(Sci.NET.LinearAlgebra.Native.CUDA PUBLIC
            Sci.NET.LinearAlgebra.Native
            ws2_32)
endif()

set_property(TARGET Sci.NET.LinearAlgebra.Native.CUDA PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded")
