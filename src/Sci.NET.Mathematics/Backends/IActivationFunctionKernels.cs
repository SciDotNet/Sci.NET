// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for activation function kernels.
/// </summary>
[PublicAPI]
public interface IActivationFunctionKernels
{
    /// <summary>
    /// Calculates the sigmoid of <paramref name="value"></paramref>.
    /// </summary>
    /// <param name="value">The value to pass to the sigmoid function.</param>
    /// <param name="result">Stores the result of the operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sigmoid<TNumber>(Scalar<TNumber> value, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Calculates the sigmoid of <paramref name="value"></paramref>.
    /// </summary>
    /// <param name="value">The value to pass to the sigmoid function.</param>
    /// <param name="result">Stores the result of the operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sigmoid<TNumber>(Tensors.Vector<TNumber> value, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Calculates the sigmoid of <paramref name="value"></paramref>.
    /// </summary>
    /// <param name="value">The value to pass to the sigmoid function.</param>
    /// <param name="result">Stores the result of the operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sigmoid<TNumber>(Matrix<TNumber> value, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Calculates the sigmoid of <paramref name="value"></paramref>.
    /// </summary>
    /// <param name="value">The value to pass to the sigmoid function.</param>
    /// <param name="result">Stores the result of the operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sigmoid<TNumber>(Tensor<TNumber> value, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Calculates the softmax of <paramref name="value"></paramref>.
    /// </summary>
    /// <param name="value">The value to pass to the softmax function.</param>
    /// <param name="result">Stores the result of the operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Softmax<TNumber>(Scalar<TNumber> value, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Calculates the softmax of <paramref name="value"></paramref>.
    /// </summary>
    /// <param name="value">The value to pass to the softmax function.</param>
    /// <param name="result">Stores the result of the operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Softmax<TNumber>(Tensors.Vector<TNumber> value, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Calculates the softmax of <paramref name="value"></paramref>.
    /// </summary>
    /// <param name="value">The value to pass to the softmax function.</param>
    /// <param name="result">Stores the result of the operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Softmax<TNumber>(Matrix<TNumber> value, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Calculates the softmax of <paramref name="value"></paramref>.
    /// </summary>
    /// <param name="value">The value to pass to the softmax function.</param>
    /// <param name="result">Stores the result of the operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Softmax<TNumber>(Tensor<TNumber> value, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;
}