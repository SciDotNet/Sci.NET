// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Common.Linq;

/// <summary>
/// Provides extension methods to find the product of a sequence.
/// </summary>
[PublicAPI]
public static class ProductExtensions
{
    /// <summary>
    /// Computes the product of a sequence.
    /// </summary>
    /// <param name="source">The sequence to find the product of.</param>
    /// <typeparam name="TNumber">The number type of the sequence.</typeparam>
    /// <returns>The inner product of the sequence.</returns>
    public static TNumber Product<TNumber>(this IEnumerable<TNumber> source)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var enumerable = source.ToList();

        return enumerable.Count == 0 ? TNumber.One : enumerable.Aggregate((a, b) => a * b);
    }
}