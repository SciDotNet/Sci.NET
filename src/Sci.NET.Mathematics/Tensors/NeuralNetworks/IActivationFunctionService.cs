// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.NeuralNetworks;

/// <summary>
/// An interface for activation function service.
/// </summary>
[PublicAPI]
public interface IActivationFunctionService
{
    /// <summary>
    /// Computes the sigmoid function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the sigmoid function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the sigmoid function.</returns>
    public ITensor<TNumber> Sigmoid<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the sigmoid function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the sigmoid derivative function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the sigmoid derivative function.</returns>
    public ITensor<TNumber> SigmoidPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the ReLU activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the ReLU function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the ReLU function.</returns>
    public ITensor<TNumber> ReLU<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the ReLU function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the ReLU derivative function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the ReLU derivative function.</returns>
    public ITensor<TNumber> ReLUPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the softmax function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the softmax function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the softmax function.</returns>
    public ITensor<TNumber> Softmax<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the softmax function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the softmax derivative function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the softmax derivative function.</returns>
    public ITensor<TNumber> SoftmaxPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>;
}