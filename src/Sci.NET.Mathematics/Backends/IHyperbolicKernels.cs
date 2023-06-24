// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for hyperbolic trigonometry kernels.
/// </summary>
[PublicAPI]
public interface IHyperbolicKernels
{
    /// <summary>
    /// Computes the hyperbolic sine of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="scalar">The <see cref="Scalar{TNumber}"/> to calculate the sinh of.</param>
    /// <param name="result">Stores the result of the operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    public void Sinh<TNumber>(Scalar<TNumber> scalar, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the hyperbolic sine of the specified <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="vector">The <see cref="Tensors.Vector{TNumber}"/> to calculate the sinh of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensors.Vector{TNumber}"/>.</typeparam>
    /// <param name="result">Stores the result of the operation.</param>
    public void Sinh<TNumber>(Tensors.Vector<TNumber> vector, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the hyperbolic sine of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="matrix">The <see cref="Matrix{TNumber}"/> to calculate the sinh of.</param>
    /// <param name="result">Stores the result of the operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    public void Sinh<TNumber>(Matrix<TNumber> matrix, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;

    /// <summary>
    /// Computes the hyperbolic sine of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Tensor{TNumber}"/> to calculate the sinh of.</param>
    /// <param name="result">Stores the result of the operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    public void Sinh<TNumber>(Tensor<TNumber> tensor, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>;
}