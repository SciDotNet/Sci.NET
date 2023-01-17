// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#include "storage/memory_block.h"

Sci::NET::BLAS::Native::MemoryBlock::MemoryBlock(const int64_t count,
                                                 void* data,
                                                 const number_type number_type)
{
    data_ptr_ = data;
    number_type_ = number_type;
    element_count_ = count;
}

int64_t Sci::NET::BLAS::Native::MemoryBlock::element_count() const
{
    return element_count_;
}

void* Sci::NET::BLAS::Native::MemoryBlock::data() const
{
    return data_ptr_;
}

Sci::NET::BLAS::Native::number_type Sci::NET::BLAS::Native::MemoryBlock::type() const
{
    return number_type_;
}
