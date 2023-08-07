// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Pointwise;

/// <summary>
/// A service for performing pointwise operations on tensors.
/// </summary>
[PublicAPI]
public interface ILinqService
{
    /// <summary>
    /// Clips the values of the <see cref="ITensor{TNumber}"/> to the specified range.
    /// </summary>
    /// <param name="tensor">The the <see cref="ITensor{TNumber}"/> to operate on.</param>
    /// <param name="min">The minimum value to clip to.</param>
    /// <param name="max">The maximum value to clip to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The clipped <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Clip<TNumber>(
        ITensor<TNumber> tensor,
        TNumber min,
        TNumber max)
        where TNumber : unmanaged, INumber<TNumber>;
}