cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(Sci.NET.Native)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/artifacts/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/artifacts/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/artifacts/lib)
set(CMAKE_INCLUDE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/artifacts/include)

#add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party/googletest/)

file(COPY ${CMAKE_CURRENT_LIST_DIR}/eng/build-props/cmake/Directory.Build.props
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

include(${CMAKE_CURRENT_LIST_DIR}/src/CMakeLists.txt)