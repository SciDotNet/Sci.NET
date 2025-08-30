// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.Intrinsics;

/// <summary>
/// Represents the SIMD instruction set available on the current platform.
/// </summary>
[Flags]
[PublicAPI]
public enum SimdInstructionSet
{
    /// <summary>
    /// No SIMD instruction set is available.
    /// </summary>
    None = 0,

    /// <summary>
    /// The SSE instruction set is available.
    /// </summary>
    Sse = 1 << 0,

    /// <summary>
    /// The SSE2 instruction set is available.
    /// </summary>
    Sse2 = 1 << 1,

    /// <summary>
    /// The SSE3 instruction set is available.
    /// </summary>
    Sse3 = 1 << 2,

    /// <summary>
    /// The SSE4.1 instruction set is available.
    /// </summary>
    Sse41 = 1 << 3,

    /// <summary>
    /// The AVX instruction set is available.
    /// </summary>
    Avx = 1 << 4,

    /// <summary>
    /// The AVX2 instruction set is available.
    /// </summary>
    Avx2 = 1 << 5,

    /// <summary>
    /// The AVX-512 instruction set is available.
    /// </summary>
    Avx512F = 1 << 6,

    /// <summary>
    /// Fused Multiply-Add (FMA) instruction set is available.
    /// </summary>
    Fma = 1 << 7
}