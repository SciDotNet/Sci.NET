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
    /// Computes the sigmoid function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the sigmoid function on.</param>
    /// <param name="result">The result of the sigmoid function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Sigmoid<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the sigmoid function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the sigmoid derivative function on.</param>
    /// <param name="result">The result of the sigmoid derivative function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void SigmoidPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the ReLU activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the ReLU function on.</param>
    /// <param name="result">The result of the ReLU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void ReLU<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the ReLU function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the ReLU derivative function on.</param>
    /// <param name="result">The result of the ReLU derivative function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void ReLUPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;
}