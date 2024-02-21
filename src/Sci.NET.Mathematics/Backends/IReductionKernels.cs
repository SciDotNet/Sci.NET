// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface providing tensor reduction operations.
/// </summary>
[PublicAPI]
public interface IReductionKernels
{
    /// <summary>
    /// Reduce a <see cref="ITensor{TNumber}"/> to a <see cref="Scalar{TNumber}"/> by adding all elements.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to sum over.</param>
    /// <param name="result">The <see cref="Scalar{TNumber}"/> to store the result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    public void ReduceAddAll<TNumber>(ITensor<TNumber> tensor, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of a <see cref="ITensor{TNumber}"/> over the specified axes.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to sum over.</param>
    /// <param name="axes">The axes to sum over.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    public void ReduceAddAxis<TNumber>(ITensor<TNumber> tensor, int[] axes, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Reduce a <see cref="ITensor{TNumber}"/> to a <see cref="Scalar{TNumber}"/> by finding the mean of all elements.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to find the mean of.</param>
    /// <param name="result">The <see cref="Scalar{TNumber}"/> to store the result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    public void ReduceMeanAll<TNumber>(ITensor<TNumber> tensor, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the mean of a <see cref="ITensor{TNumber}"/> over the specified axes.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to find the mean of.</param>
    /// <param name="axes">The axes to find the mean over.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    public void ReduceMeanAxis<TNumber>(ITensor<TNumber> tensor, int[] axes, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Reduce a <see cref="ITensor{TNumber}"/> to a <see cref="Scalar{TNumber}"/> by finding the maximum value of all elements.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to find the maximum value of.</param>
    /// <param name="result">The <see cref="Scalar{TNumber}"/> to store the result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    public void ReduceMaxAll<TNumber>(ITensor<TNumber> tensor, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the maximum value of a <see cref="ITensor{TNumber}"/> over the specified axes.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to find the maximum value of.</param>
    /// <param name="axes">The axes to find the maximum value over.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    public void ReduceMaxAxis<TNumber>(ITensor<TNumber> tensor, int[] axes, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Reduce a <see cref="ITensor{TNumber}"/> to a <see cref="Scalar{TNumber}"/> by finding the minimum value of all elements.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to find the minimum value of.</param>
    /// <param name="result">The <see cref="Scalar{TNumber}"/> to store the result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    public void ReduceMinAll<TNumber>(ITensor<TNumber> tensor, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the minimum value of a <see cref="ITensor{TNumber}"/> over the specified axes.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to find the minimum value of.</param>
    /// <param name="axes">The axes to find the minimum value over.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    public void ReduceMinAxis<TNumber>(ITensor<TNumber> tensor, int[] axes, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;
}