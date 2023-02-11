// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#include <cstdint>
#include <exception>
#include "cuda_runtime.h"
#include "storage/number_types.h"
#include "storage/memory_allocator.h"

using namespace Sci::NET::LinearAlgebra::Native;

Storage::MemoryBlock *Storage::alloc(const int64_t count,
                                     const number_type number_type) {
    void* ptr = nullptr;
    auto status = cudaMalloc(&ptr, count * sizeof_number(number_type));

    if (status != cudaSuccess) {
        throw std::exception("Unable to allocate memory.");
    }

    return new MemoryBlock(count, ptr, number_type);
}