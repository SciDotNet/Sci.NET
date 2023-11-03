project(Sci.NET.Native CXX)

include_directories(${CMAKE_CURRENT_LIST_DIR}/common/include)

include(${CMAKE_CURRENT_LIST_DIR}/Sci.NET.Native/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/Sci.NET.CUDA.Native/CMakeLists.txt)
