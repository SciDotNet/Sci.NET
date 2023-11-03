find_package(OpenMP)
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

enable_language(CUDA)
enable_testing()

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

target_compile_options(
        Sci.NET.CUDA.Native PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info>)


if (WIN32)
    target_link_libraries(Sci.NET.CUDA.Native PUBLIC
            Sci.NET.Native
            ws2_32)
endif ()


set_target_properties(Sci.NET.CUDA.Native PROPERTIES CUDA_ARCHITECTURES "50;52;53;60;61;62;70;72;75;80;86;90")

set_property(TARGET Sci.NET.CUDA.Native PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded")

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

file(GLOB_RECURSE TEST_SRC_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/tests/*.c
        ${CMAKE_CURRENT_LIST_DIR}/tests/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/tests/*.cpp)

file(GLOB_RECURSE TEST_HEADER_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/*.h
        ${CMAKE_CURRENT_LIST_DIR}/src/*.hpp)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(Sci.NET.CUDA.Native.Tests
        ${TEST_SRC_FILES}
        ${TEST_HEADER_FILES})

target_link_libraries(Sci.NET.CUDA.Native.Tests PUBLIC
        Sci.NET.CUDA.Native
        gtest_main
        gmock_main
        gtest
        gmock)