// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;
using Sci.NET.Common.Numerics;

namespace Sci.NET.Common.Performance;

/// <summary>
/// Provides guards for vectorization.
/// </summary>
[PublicAPI]
public static class VectorGuard
{
    /// <summary>
    /// Determines whether vectorization is possible for the given type.
    /// </summary>
    /// <typeparam name="TNumber">The type to check.</typeparam>
    /// <returns>True if vectorization is possible, false otherwise.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool CanVectorize<TNumber>()
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (typeof(TNumber) == typeof(BFloat16))
        {
            return false;
        }

        return Vector.IsHardwareAccelerated && Vector<TNumber>.Count > 1 && Unsafe.SizeOf<TNumber>() < 8;
    }
}