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
    /// Reduces a <see cref="ITensor{TNumber}"/> to a given shape.
    /// </summary>
    /// <param name="tensor">Reduces the <see cref="ITensor{TNumber}"/> to the given shape.</param>
    /// <param name="targetShape">The shape to reduce the <see cref="ITensor{TNumber}"/> to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The reduced <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> ReduceToShape<TNumber>(ITensor<TNumber> tensor, Shape targetShape)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Determines if a <see cref="ITensor{TNumber}"/> can be reduced to a given shape.
    /// </summary>
    /// <param name="tensor">Reduces the <see cref="ITensor{TNumber}"/> to the given shape.</param>
    /// <param name="shape">The shape to reduce the <see cref="ITensor{TNumber}"/> to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>A value indicating whether the <see cref="ITensor{TNumber}"/> can be reduced to the given shape.</returns>
    public bool CanReduceToShape<TNumber>(ITensor<TNumber> tensor, Shape shape)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the sum of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to sum.</param>
    /// <param name="axes">The axes to sum over.</param>
    /// <param name="keepDims">A value indicating whether the dimensions should be preserved.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The sum of all of the elements in the <see cref="ITensor{TNumber}"/> over the selected axes.</returns>
    public ITensor<TNumber> Sum<TNumber>(ITensor<TNumber> tensor, int[]? axes = null, bool keepDims = false)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the mean of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to find the mean of.</param>
    /// <param name="axes">The axes to find the mean over.</param>
    /// <param name="keepDims">A value indicating whether the dimensions should be preserved.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The mean of all of the elements in the <see cref="ITensor{TNumber}"/> over the selected axes.</returns>
    public ITensor<TNumber> Mean<TNumber>(ITensor<TNumber> tensor, int[]? axes = null, bool keepDims = false)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the maximum value of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to find the maximum value of.</param>
    /// <param name="axes">The axes to find the maximum value over.</param>
    /// <param name="keepDims">A value indicating whether the dimensions should be preserved.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The maximum value of all of the elements in the <see cref="ITensor{TNumber}"/> over the selected axes.</returns>
    public ITensor<TNumber> Max<TNumber>(ITensor<TNumber> tensor, int[]? axes = null, bool keepDims = false)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the minimum value of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to find the minimum value of.</param>
    /// <param name="axes">The axes to find the minimum value over.</param>
    /// <param name="keepDims">A value indicating whether the dimensions should be preserved.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The minimum value of all of the elements in the <see cref="ITensor{TNumber}"/> over the selected axes.</returns>
    public ITensor<TNumber> Min<TNumber>(ITensor<TNumber> tensor, int[]? axes = null, bool keepDims = false)
        where TNumber : unmanaged, INumber<TNumber>;
}