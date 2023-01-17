// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#include <cstdint>
#include "storage/number_types.h"
#include "storage/memory_allocator.h"

Sci::NET::BLAS::Native::MemoryBlock* Sci::NET::BLAS::Native::Storage::alloc(const int64_t count,
                                                                            const number_type number_type)
{
    const auto data_ptr = calloc(count, sizeof_number(number_type));

    return new MemoryBlock(count, data_ptr, number_type);
}
