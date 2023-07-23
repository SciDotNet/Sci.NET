// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for a backend that provides power operations.
/// </summary>
[PublicAPI]
public interface IPowerKernels
{
    /// <summary>
    /// Raises <paramref name="value"/> to the power of <paramref name="power"/> and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <param name="result">The result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Pow<TNumber>(Scalar<TNumber> value, Scalar<TNumber> power, Scalar<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Raises <paramref name="value"/> to the power of <paramref name="power"/> and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <param name="result">The result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Pow<TNumber>(Tensors.Vector<TNumber> value, Scalar<TNumber> power, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Raises <paramref name="value"/> to the power of <paramref name="power"/> and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <param name="result">The result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Pow<TNumber>(Matrix<TNumber> value, Scalar<TNumber> power, Matrix<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Raises <paramref name="value"/> to the power of <paramref name="power"/> and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <param name="result">The result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Pow<TNumber>(Tensor<TNumber> value, Scalar<TNumber> power, Tensor<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Raises a <see cref="Scalar{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result in.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Square<TNumber>(Scalar<TNumber> value, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Raises a <see cref="Tensors.Vector{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result in.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Square<TNumber>(Tensors.Vector<TNumber> value, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Raises a <see cref="Matrix{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result in.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Square<TNumber>(Matrix<TNumber> value, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Raises a <see cref="Tensor{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result in.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Square<TNumber>(Tensor<TNumber> value, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Raises e to the power of <paramref name="value"/> and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="value">The exponent.</param>
    /// <param name="result">The <see cref="Scalar{TNumber}"/> to store the result in.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Exp<TNumber>(Scalar<TNumber> value, Scalar<TNumber> result)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Raises e to the power of <paramref name="value"/> and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="value">The exponent.</param>
    /// <param name="result">The <see cref="Tensors.Vector{TNumber}"/> to store the result in.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Exp<TNumber>(Tensors.Vector<TNumber> value, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Raises e to the power of <paramref name="value"/> and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="value">The exponent.</param>
    /// <param name="result">The <see cref="Matrix{TNumber}"/> to store the result in.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Exp<TNumber>(Matrix<TNumber> value, Matrix<TNumber> result)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Raises e to the power of <paramref name="value"/> and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="value">The exponent.</param>
    /// <param name="result">The <see cref="Tensor{TNumber}"/> to store the result in.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Exp<TNumber>(Tensor<TNumber> value, Tensor<TNumber> result)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>;
}