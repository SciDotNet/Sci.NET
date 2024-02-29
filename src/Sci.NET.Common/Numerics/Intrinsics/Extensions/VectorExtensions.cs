// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Common.Numerics.Intrinsics.Extensions;

/// <summary>
/// Extensions for <see cref="ISimdVector{TNumber}"/>.
/// </summary>
[PublicAPI]
public static class VectorExtensions
{
    /// <summary>
    /// Raises <see cref="IFloatingPointConstants{TNumber}.E"/> to the power of <paramref name="vector"/>.
    /// </summary>
    /// <param name="vector">The vector to operate on.</param>
    /// <typeparam name="TNumber">The number type of the vector.</typeparam>
    /// <returns>The result of the exp operation.</returns>
    public static ISimdVector<TNumber> Exp<TNumber>(this ISimdVector<TNumber> vector)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        Span<TNumber> values = stackalloc TNumber[vector.Count];

        for (var i = 0; i < vector.Count; i++)
        {
            values[i] = TNumber.Exp(vector[i]);
        }

        return vector.CreateWith(values);
    }
}