// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Attributes;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for linear algebra backends.
/// </summary>
[PublicAPI]
public interface ILinearAlgebraKernels
{
    /// <summary>
    /// Multiplies two matrices.
    /// </summary>
    /// <param name="left">The left matrix.</param>
    /// <param name="right">The right matrix.</param>
    /// <param name="result">The result of the matrix multiplication.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    [AssumesValidShape]
    [AssumesValidDevice]
    public void MatrixMultiply<TNumber>(
        [AssumesShape("i,-1")] Matrix<TNumber> left,
        [AssumesShape("-1,j")] Matrix<TNumber> right,
        [AssumesShape("i, j")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the inner product of two vectors.
    /// </summary>
    /// <param name="left">The left vector.</param>
    /// <param name="right">The right vector.</param>
    /// <param name="result">The result of the inner product.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensors.Vector{TNumber}"/>.</typeparam>
    [AssumesValidShape]
    public void InnerProduct<TNumber>(
        [AssumesShape("i")] Tensors.Vector<TNumber> left,
        [AssumesShape("i")] Tensors.Vector<TNumber> right,
        Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;
}