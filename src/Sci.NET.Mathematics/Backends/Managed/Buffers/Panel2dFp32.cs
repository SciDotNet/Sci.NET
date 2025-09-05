// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.Backends.Managed.Buffers;

internal readonly struct Panel2dFp32
{
    public readonly unsafe float* A;
    public readonly unsafe float* B;

    public unsafe Panel2dFp32(float* a, float* b)
    {
        A = a;
        B = b;
    }
}