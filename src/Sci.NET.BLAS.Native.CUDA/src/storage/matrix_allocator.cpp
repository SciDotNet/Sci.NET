// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#include <cuda_runtime.h>
#include "storage/matrix_allocator.h"

Sci::NET::BLAS::Native::Matrix *Sci::NET::BLAS::Native::Storage::alloc_matrix(int64_t size_x, int64_t size_y,
                                                                                   Sci::NET::BLAS::Native::matrixNumberType type) {
    auto size = size_x * size_y * sizeof_number(type);
    void* dataPtr;
    auto status = cudaMalloc(&dataPtr, size);

    if (status != cudaSuccess) {
        return nullptr;
    }
    
    return new Matrix(size_x, size_y, dataPtr, type);
}
