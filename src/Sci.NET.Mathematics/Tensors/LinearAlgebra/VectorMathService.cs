// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.LinearAlgebra;

/// <summary>
/// An interface for <see cref="Vector{TNumber}"/> operations.
/// </summary>
[PublicAPI]
public interface IVectorOperationsService
{
    /// <summary>
    /// Computes the cosine distance between two <see cref="Vector{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="Vector{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Vector{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>s.</typeparam>
    /// <returns>The cosine distance between the two <see cref="Vector{TNumber}"/>s.</returns>
    public Scalar<TNumber> CosineDistance<TNumber>(Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, IFloatingPoint<TNumber>, IRootFunctions<TNumber>;

    /// <summary>
    /// Computes the norm of a <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="vector">The <see cref="Vector{TNumber}"/> to compute the norm of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The norm of the <see cref="Vector{TNumber}"/>.</returns>
    public Scalar<TNumber> Norm<TNumber>(Vector<TNumber> vector)
        where TNumber : unmanaged, IRootFunctions<TNumber>, IFloatingPoint<TNumber>;
}