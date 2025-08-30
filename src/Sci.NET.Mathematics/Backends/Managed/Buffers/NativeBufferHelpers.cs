// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.Intrinsics.X86;

namespace Sci.NET.Mathematics.Backends.Managed.Buffers;

internal static class NativeBufferHelpers
{
    public const int L1CacheSize = 32 * 1024; // A typical L1 cache size in bytes
    public const int AvxVectorSizeFp32 = 8;
    public const int AvxVectorSizeFp64 = 4;
    public const int L1Size = 32 * 1024;
    public const int TileSizeFp32 = L1Size / sizeof(float);
    public const int TileSizeFp64 = L1Size / sizeof(double);

    public static unsafe void Pack1dFp32Avx(float* src, float* dst, long n)
    {
        long i = 0;
        for (; i <= n - AvxVectorSizeFp32; i += AvxVectorSizeFp32)
        {
            var vec = Avx.LoadVector256(src + i);
            Avx.Store(dst + i, vec);
        }

        for (; i < n; ++i)
        {
            dst[i] = src[i];
        }
    }

    public static unsafe void Pack1dFp64Avx(double* src, double* dst, long n)
    {
        long i = 0;
        for (; i <= n - AvxVectorSizeFp64; i += AvxVectorSizeFp64)
        {
            var vec = Avx.LoadVector256(src + i);
            Avx.Store(dst + i, vec);
        }

        for (; i < n; ++i)
        {
            dst[i] = src[i];
        }
    }
}