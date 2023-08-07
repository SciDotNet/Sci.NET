file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/*.c
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/*.h
        ${CMAKE_CURRENT_LIST_DIR}/src/*.hpp)

file(GLOB_RECURSE PUBLIC_HEADER_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/include/*.h
        ${CMAKE_CURRENT_LIST_DIR}/include/*.hpp)

add_library(Sci.NET.Native STATIC
        ${SRC_FILES}
        ${HEADER_FILES}
        ${PUBLIC_HEADER_FILES})

set_property(TARGET Sci.NET.Native PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded")

target_include_directories(Sci.NET.Native PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
