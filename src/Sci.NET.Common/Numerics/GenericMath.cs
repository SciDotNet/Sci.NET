// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

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
        var epsilon = float.CreateChecked(TNumber.Zero) + 0.4f;
        return TNumber.CreateChecked(epsilon) != TNumber.Zero;
    }
}