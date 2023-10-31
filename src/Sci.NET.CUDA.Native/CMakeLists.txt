find_package(OpenMP)
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

enable_language(CUDA)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/*.c
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

file(GLOB_RECURSE CUDA_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cu)

file(GLOB_RECURSE CUDA_HEADER_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cuh)

file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/*.h
        ${CMAKE_CURRENT_LIST_DIR}/src/*.hpp)

add_library(Sci.NET.CUDA.Native SHARED
        ${SRC_FILES}
        ${HEADER_FILES}
        ${CUDA_FILES}
        ${CUDA_HEADER_FILES})

target_include_directories(Sci.NET.Native PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

target_link_libraries(Sci.NET.CUDA.Native PUBLIC
        OpenMP::OpenMP_CXX
        CUDA::cudart_static
        CUDA::cublas
        CUDA::curand
        CUDA::cusolver
        CUDA::cusparse
        CUDA::nvJitLink_static
        CUDA::nvrtc_static)


if (WIN32)
    target_link_libraries(Sci.NET.CUDA.Native PUBLIC
            Sci.NET.Native
            ws2_32)
endif ()


set_target_properties(Sci.NET.CUDA.Native PROPERTIES CUDA_ARCHITECTURES "50;52;53;60;61;62;70;72;75;80;86;90")

set_property(TARGET Sci.NET.CUDA.Native PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded")
