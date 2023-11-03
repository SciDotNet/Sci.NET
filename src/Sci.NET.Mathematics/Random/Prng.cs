// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using Sci.NET.Common.Memory;

namespace Sci.NET.Mathematics.Random;

/// <summary>
/// Random number generation.
/// </summary>
[PublicAPI]
[SuppressMessage("Security", "CA5394:Do not use insecure randomness", Justification = "This is not for security purposes.")]
public static class Prng
{
    /// <summary>
    /// Generates a random number.
    /// </summary>
    /// <param name="block">The block to fill with random numbers.</param>
    /// <param name="min">The minimum value of the distribution.</param>
    /// <param name="max">The maximum value of the distribution.</param>
    /// <param name="seed">The seed for the random number generator.</param>
    /// <typeparam name="TNumber">The type of the number to generate.</typeparam>
    /// <exception cref="NotSupportedException">Thrown if the number type is not supported.</exception>
    public static void Uniform<TNumber>(
        IMemoryBlock<TNumber> block,
        TNumber min,
        TNumber max,
        int? seed = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (block is not SystemMemoryBlock<TNumber> systemBlock)
        {
            throw new NotSupportedException("Only SystemMemoryBlock is supported.");
        }

        using var rng = new DeterministicRandomNumberGenerator(seed);

        _ = Parallel.For(0, block.Length, i => systemBlock[i] = rng.Next(min, max));
    }
}