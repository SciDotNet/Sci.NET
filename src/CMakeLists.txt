project(Sci.NET.Native CXX)

include_directories(${CMAKE_CURRENT_LIST_DIR}/common/include)

set(ADD_CXX_SRC_FILES ${CMAKE_SOURCE_DIR}/eng/build-props/cmake/AddSourceFiles.cmake)
set(ADD_CUDA_SRC_FILES ${CMAKE_SOURCE_DIR}/eng/build-props/cmake/AddCudaSourceFiles.cmake)

include(${CMAKE_CURRENT_LIST_DIR}/Sci.NET.LinearAlgebra.Native/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/Sci.NET.LinearAlgebra.Native.CPU/CMakeLists.txt)
#include(${CMAKE_CURRENT_LIST_DIR}/Sci.NET.LinearAlgebra.Native.CUDA/CMakeLists.txt)