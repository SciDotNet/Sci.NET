// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.Backends.Managed.Buffers;

internal readonly struct Panel2dFp64
{
    public readonly unsafe double* A;
    public readonly unsafe double* B;

    public unsafe Panel2dFp64(double* a, double* b)
    {
        B = b;
        A = a;
    }
}