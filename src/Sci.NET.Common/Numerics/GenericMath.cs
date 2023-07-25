// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;

namespace Sci.NET.Common.Numerics;

/// <summary>
/// A helper class for generic math operations.
/// </summary>
[PublicAPI]
public static class GenericMath
{
    /// <summary>
    /// Determines if the number type is floating point.
    /// </summary>
    /// <typeparam name="TNumber">The number type to test.</typeparam>
    /// <returns><c>true</c> if the number is a floating point type, else, <c>false</c>.</returns>
    public static bool IsFloatingPoint<TNumber>()
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TNumber.CreateChecked(0.01f) != TNumber.Zero;
    }

    /// <summary>
    /// Gets the machine epsilon for the specified number type.
    /// </summary>
    /// <typeparam name="TNumber">The number type to get the epsilon for.</typeparam>
    /// <returns>The machine epsilon for the specified number type.</returns>
    public static unsafe TNumber Epsilon<TNumber>()
        where TNumber : unmanaged, INumber<TNumber>
    {
        var numBytes = Unsafe.SizeOf<TNumber>();
        var bytes = stackalloc byte[Unsafe.SizeOf<TNumber>()];

        if (BitConverter.IsLittleEndian)
        {
            bytes[0] = 0x01;
        }
        else
        {
            bytes[numBytes - 1] = 0x01;
        }

        return Unsafe.Read<TNumber>(bytes);
    }
}