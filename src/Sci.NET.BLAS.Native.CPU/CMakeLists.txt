find_package(OpenMP)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/*.c
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/*.h
        ${CMAKE_CURRENT_LIST_DIR}/src/*.hpp)

add_library(Sci.NET.BLAS.Native.CPU SHARED
        ${SRC_FILES}
        ${HEADER_FILES})

target_link_libraries(Sci.NET.BLAS.Native.CPU PRIVATE 
        Sci.NET.BLAS.Native
        OpenMP::OpenMP_CXX)

set_property(TARGET Sci.NET.BLAS.Native.CPU PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded")
