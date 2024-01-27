// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Reduction;

/// <summary>
/// Provides reduction operations for <see cref="ITensor{TNumber}"/>s.
/// </summary>
[PublicAPI]
public interface IReductionService
{
    /// <summary>
    /// Computes the sum of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to sum.</param>
    /// <param name="axes">The axes to sum over.</param>
    /// <param name="keepDims">A value indicating whether the dimensions should be preserved.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The sum of all of the elements in the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Sum<TNumber>(ITensor<TNumber> tensor, int[]? axes = null, bool keepDims = false)
        where TNumber : unmanaged, INumber<TNumber>;
}