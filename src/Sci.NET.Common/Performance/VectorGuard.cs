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
    /// <param name="count">The vector count.</param>
    /// <returns>True if vectorization is possible, false otherwise.</returns>
    [MethodImpl(ImplementationOptions.HotPath)]
    public static bool IsSupported<TNumber>(out int count)
        where TNumber : unmanaged
    {
        if (typeof(TNumber) == typeof(BFloat16))
        {
            count = 0;
            return false;
        }

        if (!Vector<TNumber>.IsSupported)
        {
            count = 0;
            return false;
        }

        count = Vector<TNumber>.Count;
        return Vector.IsHardwareAccelerated && Unsafe.SizeOf<TNumber>() < 8;
    }
}