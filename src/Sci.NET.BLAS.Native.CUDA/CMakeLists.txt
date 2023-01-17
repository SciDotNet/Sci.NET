find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/*.c
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cu)

file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/*.h
        ${CMAKE_CURRENT_LIST_DIR}/src/*.hpp
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cuh)

add_library(Sci.NET.BLAS.Native.CUDA SHARED
        ${SRC_FILES}
        ${HEADER_FILES})

target_link_libraries(Sci.NET.BLAS.Native.CUDA PRIVATE
        Sci.NET.BLAS.Native
        CUDA::cudart_static)