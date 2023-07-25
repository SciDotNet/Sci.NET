// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.Random;

/// <summary>
/// Extension methods for randomization.
/// </summary>
[PublicAPI]
public static class RandomExtensions
{
    /// <summary>
    /// Shuffles a collection.
    /// </summary>
    /// <param name="collection">The collection to shuffle.</param>
    /// <param name="seed">The seed to use for the random number generator.</param>
    /// <typeparam name="T">The type of the collection elements.</typeparam>
    public static void Shuffle<T>(this IList<T> collection, int? seed = null)
    {
        _ = seed;
        var random = new System.Random();

        var n = collection.Count;

        while (n > 1)
        {
            n--;
#pragma warning disable CA5394
            var k = random.Next(0, n + 1);
#pragma warning restore CA5394
            (collection[k], collection[n]) = (collection[n], collection[k]);
        }
    }
}