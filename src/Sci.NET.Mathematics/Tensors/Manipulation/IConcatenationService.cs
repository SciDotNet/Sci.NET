// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Manipulation;

/// <summary>
/// A service for concatenating tensors.
/// </summary>
[PublicAPI]
public interface IConcatenationService
{
    /// <summary>
    /// Concatenates a collection of <see cref="Scalar{TNumber}"/> into a <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="scalars">The scalars to concatenate.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The concatenated <see cref="Scalar{TNumber}"/> collection.</returns>
    public Vector<TNumber> Concatenate<TNumber>(ICollection<Scalar<TNumber>> scalars)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Concatenates a collection of <see cref="Vector{TNumber}"/> into a <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="vectors">The vectors to concatenate.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The concatenated <see cref="Vector{TNumber}"/> collection.</returns>
    public Matrix<TNumber> Concatenate<TNumber>(ICollection<Vector<TNumber>> vectors)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Concatenates a collection of <see cref="Matrix{TNumber}"/> into a <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="matrices">The matrices to concatenate.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The concatenated <see cref="Matrix{TNumber}"/> collection.</returns>
    public Tensor<TNumber> Concatenate<TNumber>(ICollection<Matrix<TNumber>> matrices)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Concatenates a collection of <see cref="Tensor{TNumber}"/> into a <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensors">The tensors to concatenate.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The concatenated <see cref="Tensor{TNumber}"/> collection.</returns>
    public Tensor<TNumber> Concatenate<TNumber>(ICollection<Tensor<TNumber>> tensors)
        where TNumber : unmanaged, INumber<TNumber>;
}