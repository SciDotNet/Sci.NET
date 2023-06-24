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
    /// Reduce an <see cref="ITensor{TNumber}"/> to a <see cref="ITensor{TNumber}"/> by adding all elements, but keeping
    /// the existing dimensions.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to sum over.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    public void ReduceAddAllKeepDims<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Reduce a <see cref="ITensor{TNumber}"/> to a <see cref="Scalar{TNumber}"/> by adding all elements along the given
    /// <paramref name="axes"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to sum over.</param>
    /// <param name="axes">The axes to sum over.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    public void ReduceAddAxis<TNumber>(ITensor<TNumber> tensor, int[] axes, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Reduce a <see cref="ITensor{TNumber}"/> to a <see cref="ITensor{TNumber}"/> by adding all elements along the given
    /// <paramref name="axes"/> but keeping the existing dimensions.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to sum over.</param>
    /// <param name="axes">The axes to sum over.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    public void ReduceAddAxisKeepDims<TNumber>(ITensor<TNumber> tensor, int[] axes, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;
}