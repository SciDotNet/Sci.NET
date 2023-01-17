// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#include "c_api.h"
#include "storage/matrix_allocator.h"

apiStatusCode_t allocMatrix(int64_t x, int64_t y, matrixNumberType type, Matrix **result) {
    auto* matrix = Storage::alloc_matrix(x, y, type);
    *result = matrix;
    return sdnSuccess;
}

apiStatusCode_t matmul(Matrix *left, Matrix *right, Matrix **result) {
    return sdnStatusUnknown;
}

apiStatusCode_t setData(Matrix *destination, void *source, int64_t length) {
    return sdnStatusUnknown;
}

apiStatusCode_t getData(Matrix *source, void *destination, int64_t length) {
    return sdnStatusUnknown;
}
