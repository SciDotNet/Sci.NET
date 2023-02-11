// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#include "storage/memory_block.h"

using namespace Sci::NET::LinearAlgebra::Native;

Storage::MemoryBlock::MemoryBlock(const int64_t count,
                                  void *data,
                                  const number_type number_type) :
        element_count_(count),
        data_ptr_(data),
        number_type_(number_type) {
}

int64_t Storage::MemoryBlock::element_count() const {
    return element_count_;
}

void *Storage::MemoryBlock::data() const {
    return data_ptr_;
}

number_type Storage::MemoryBlock::type() const {
    return number_type_;
}

int64_t Storage::MemoryBlock::data_size() const {
    return element_count_ * (std::int64_t)Native::sizeof_number(number_type_);
}
