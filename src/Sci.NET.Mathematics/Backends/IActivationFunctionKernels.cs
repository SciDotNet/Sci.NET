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
    public void SigmoidBackard<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
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
    public void ReLUBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the softmax activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the softmax function on.</param>
    /// <param name="sumBuffer">The buffer to store the sum of the exponential scores.</param>
    /// <param name="result">The result of the softmax function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Softmax<TNumber>(ITensor<TNumber> value, Scalar<TNumber> sumBuffer, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the softmax function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the softmax derivative function on.</param>
    /// <param name="softmaxValue">The result of the softmax function.</param>
    /// <param name="result">The result of the softmax derivative function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void SoftmaxBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> softmaxValue, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the Leaky ReLU activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Leaky ReLU function on.</param>
    /// <param name="result">The result of the Leaky ReLU function.</param>
    /// <param name="alpha">The alpha value for the Leaky ReLU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void LeakyReLU<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the Leaky ReLU function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Leaky ReLU derivative function on.</param>
    /// <param name="result">The result of the Leaky ReLU derivative function.</param>
    /// <param name="alpha">The alpha value for the Leaky ReLU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void LeakyReLUBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the ELU activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the ELU function on.</param>
    /// <param name="result">The result of the ELU function.</param>
    /// <param name="alpha">The alpha value for the ELU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Elu<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the ELU function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the ELU derivative function on.</param>
    /// <param name="result">The result of the ELU derivative function.</param>
    /// <param name="alpha">The alpha value for the ELU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void EluBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Computes the CELU activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the CELU function on.</param>
    /// <param name="result">The result of the CELU function.</param>
    /// <param name="alpha">The alpha value for the CELU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Celu<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the CELU function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the CELU derivative function on.</param>
    /// <param name="result">The result of the CELU derivative function.</param>
    /// <param name="alpha">The alpha value for the CELU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void CeluBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Computes the Swish activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Swish function on.</param>
    /// <param name="result">The result of the Swish function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Swish<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the Swish function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Swish derivative function on.</param>
    /// <param name="result">The result of the Swish derivative function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void SwishBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the Mish activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Mish function on.</param>
    /// <param name="result">The result of the Mish function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Mish<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IHyperbolicFunctions<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the Mish function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Mish derivative function on.</param>
    /// <param name="result">The result of the Mish derivative function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void MishBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IHyperbolicFunctions<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the hard Tanh activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the hard Tanh function on.</param>
    /// <param name="result">The result of the hard Tanh function.</param>
    /// <param name="min">The minimum value for the hard Tanh function.</param>
    /// <param name="max">The maximum value for the hard Tanh function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void HardTanh<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the hard Tanh function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the hard Tanh derivative function on.</param>
    /// <param name="result">The result of the hard Tanh derivative function.</param>
    /// <param name="min">The minimum value for the hard Tanh function.</param>
    /// <param name="max">The maximum value for the hard Tanh function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void HardTanhBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the hard Sigmoid activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the hard Sigmoid function on.</param>
    /// <param name="result">The result of the hard Sigmoid function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void HardSigmoid<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the hard Sigmoid function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the hard Sigmoid derivative function on.</param>
    /// <param name="result">The result of the hard Sigmoid derivative function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void HardSigmoidBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the log sigmoid activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the log sigmoid function on.</param>
    /// <param name="result">The result of the log sigmoid function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void LogSigmoid<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the log sigmoid function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the log sigmoid derivative function on.</param>
    /// <param name="result">The result of the log sigmoid derivative function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void LogSigmoidBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the GELU activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the GELU function on.</param>
    /// <param name="result">The result of the GELU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void GELU<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the GELU function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the GELU derivative function on.</param>
    /// <param name="result">The result of the GELU derivative function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void GELUBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the softplus activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the softplus function on.</param>
    /// <param name="result">The result of the softplus function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void SoftPlus<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the softplus function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the softplus derivative function on.</param>
    /// <param name="result">The result of the softplus derivative function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void SoftPlusBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the softsign activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the softsign function on.</param>
    /// <param name="result">The result of the softsign function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void SoftSign<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the softsign function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the softsign derivative function on.</param>
    /// <param name="result">The result of the softsign derivative function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void SoftSignBackward<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;
}