// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#ifndef SCI_NET_MATRIX_ALLOCATOR_H
#define SCI_NET_MATRIX_ALLOCATOR_H

#include <cstdint>
#include <cstdlib>
#include "memory_block.h"

namespace Sci::NET::LinearAlgebra::Native::Storage {
    MemoryBlock* alloc(int64_t count, number_type number_type);
}

#endif //SCI_NET_MATRIX_ALLOCATOR_H