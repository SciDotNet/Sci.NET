// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#ifndef SCI_NET_MATRIX_NUMBER_TYPES_CPP
#define SCI_NET_MATRIX_NUMBER_TYPES_CPP

#include "storage/number_types.h"


size_t Sci::NET::BLAS::Native::sizeof_number(const number_type type) {
    switch (type) {
        case float32:
            return sizeof(float);
        case float64:
            return sizeof(double);
    }

    return sizeof(unsigned char);
}

#endif //SCI_NET_MATRIX_NUMBER_TYPES_CPP