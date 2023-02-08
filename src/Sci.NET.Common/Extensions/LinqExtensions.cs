// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Common.Extensions;

/// <summary>
/// Custom LINQ extensions.
/// </summary>
[PublicAPI]
public static class LinqExtensions
{
    /// <summary>
    /// Returns the product of a sequence of values.
    /// </summary>
    /// <param name="source">The source array.</param>
    /// <typeparam name="T">The type of the array.</typeparam>
    /// <returns>The product of the sequence of values.</returns>
    public static T Product<T>(this IEnumerable<T> source)
        where T : INumber<T>
    {
        return source.Aggregate(T.One, (current, item) => current * item);
    }
}