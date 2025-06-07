﻿// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.Intrinsics.X86;

namespace Sci.NET.Common.Intrinsics;

/// <summary>
/// A helper class for SIMD intrinsics.
/// </summary>
[PublicAPI]
public static class IntrinsicsHelper
{
    /// <summary>
    /// Gets the available SIMD instruction sets on the current platform.
    /// </summary>
    public static readonly SimdInstructionSet AvailableInstructionSets;

#pragma warning disable CA1810
    static IntrinsicsHelper()
#pragma warning restore CA1810
    {
        AvailableInstructionSets = SimdInstructionSet.None;

        if (Sse.IsSupported)
        {
            AvailableInstructionSets |= SimdInstructionSet.Sse;
        }

        if (Sse2.IsSupported)
        {
            AvailableInstructionSets |= SimdInstructionSet.Sse2;
        }

        if (Sse3.IsSupported)
        {
            AvailableInstructionSets |= SimdInstructionSet.Sse3;
        }

        if (Sse41.IsSupported)
        {
            AvailableInstructionSets |= SimdInstructionSet.Sse41;
        }

        if (Avx.IsSupported)
        {
            AvailableInstructionSets |= SimdInstructionSet.Avx;
        }

        if (Avx2.IsSupported)
        {
            AvailableInstructionSets |= SimdInstructionSet.Avx2;
        }

        if (Avx512F.IsSupported)
        {
            AvailableInstructionSets |= SimdInstructionSet.Avx512F;
        }
    }

    /// <summary>
    /// Gets the required alignment for SIMD operations based on the available instruction sets.
    /// </summary>
    /// <returns>The required alignment as a <see cref="UIntPtr"/>.</returns>
    public static UIntPtr CalculateRequiredAlignment()
    {
        if (Avx512F.IsSupported)
        {
            return new UIntPtr(64);
        }

        if (Avx2.IsSupported || Avx.IsSupported)
        {
            return new UIntPtr(32);
        }

        if (Sse41.IsSupported || Sse3.IsSupported || Sse2.IsSupported || Sse.IsSupported)
        {
            return new UIntPtr(16);
        }

        return new UIntPtr(1);
    }
}