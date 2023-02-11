// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Backends;

/// <summary>
/// An interface for random operations for an <see cref="ITensor{TNumber}"/>.
/// </summary>
[PublicAPI]
public interface IRandomBackendOperations
{
    /// <summary>
    /// Creates an <see cref="ITensor{TNumber}"/> filled with random values from
    /// the specified generator.
    /// </summary>
    /// <param name="shape">The shape of the <see cref="ITensor{TNumber}"/> to create.</param>
    /// <param name="min">The minimum value to be generated.</param>
    /// <param name="max">The maximum value to be generated.</param>
    /// <param name="seed">The random seed.</param>
    /// <typeparam name="TNumber">The type of number to be generated.</typeparam>
    /// <returns>A new <see cref="ITensor{TNumber}"/> filled with random data.</returns>
    public ITensor<TNumber> Uniform<TNumber>(Shape shape, TNumber min, TNumber max, long seed)
        where TNumber : unmanaged, INumber<TNumber>;
}