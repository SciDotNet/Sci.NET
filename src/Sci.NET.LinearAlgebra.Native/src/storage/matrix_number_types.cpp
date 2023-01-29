// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#ifndef SCI_NET_MATRIX_NUMBER_TYPES_CPP
#define SCI_NET_MATRIX_NUMBER_TYPES_CPP

#include "storage/number_types.h"

size_t Sci::NET::LinearAlgebra::Native::sizeof_number(const number_type type) {
    switch (type) {
        case number_type::float32:
            return sizeof(float);
        case number_type::float64:
            return sizeof(double);
        case number_type::uint8:
            return sizeof(unsigned char);
    }

    return sizeof(unsigned char);
}

#endif //SCI_NET_MATRIX_NUMBER_TYPES_CPP