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
    public ITensor<TNumber> SigmoidBackward<TNumber>(ITensor<TNumber> value)
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
    public ITensor<TNumber> ReLUBackward<TNumber>(ITensor<TNumber> value)
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
    public ITensor<TNumber> SoftmaxBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the Leaky ReLU activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Leaky ReLU function on.</param>
    /// <param name="alpha">The alpha value for the Leaky ReLU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the Leaky ReLU function.</returns>
    public ITensor<TNumber> LeakyReLU<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the Leaky ReLU function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Leaky ReLU derivative function on.</param>
    /// <param name="alpha">The alpha value for the Leaky ReLU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the Leaky ReLU derivative function.</returns>
    public ITensor<TNumber> LeakyReLUBackward<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the ELU activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the ELU function on.</param>
    /// <param name="alpha">The alpha value for the ELU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the ELU function.</returns>
    public ITensor<TNumber> Elu<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the ELU function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the ELU derivative function on.</param>
    /// <param name="alpha">The alpha value for the ELU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the ELU derivative function.</returns>
    public ITensor<TNumber> EluBackward<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Computes the CELU activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the CELU function on.</param>
    /// <param name="alpha">The alpha value for the CELU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the CELU function.</returns>
    public ITensor<TNumber> Celu<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the CELU function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the CELU derivative function on.</param>
    /// <param name="alpha">The alpha value for the CELU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the CELU derivative function.</returns>
    public ITensor<TNumber> CeluBackward<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Computes the Swish activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Swish function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the Swish function.</returns>
    public ITensor<TNumber> Swish<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the Swish function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Swish derivative function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the Swish derivative function.</returns>
    public ITensor<TNumber> SwishBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Computes the Mish activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Mish function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the Mish function.</returns>
    public ITensor<TNumber> Mish<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IHyperbolicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the Mish function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Mish derivative function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the Mish derivative function.</returns>
    public ITensor<TNumber> MishBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IHyperbolicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Computes the hard Tanh activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the hard Tanh function on.</param>
    /// <param name="min">The minimum value of the hard Tanh function.</param>
    /// <param name="max">The maximum value of the hard Tanh function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the hard Tanh function.</returns>
    public ITensor<TNumber> HardTanh<TNumber>(ITensor<TNumber> value, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the hard Tanh function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the hard Tanh derivative function on.</param>
    /// <param name="min">The minimum value of the hard Tanh function.</param>
    /// <param name="max">The maximum value of the hard Tanh function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the hard Tanh derivative function.</returns>
    public ITensor<TNumber> HardTanhBackward<TNumber>(ITensor<TNumber> value, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the hard sigmoid activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the hard sigmoid function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the hard sigmoid function.</returns>
    public ITensor<TNumber> HardSigmoid<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the hard sigmoid function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the hard sigmoid derivative function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the hard sigmoid derivative function.</returns>
    public ITensor<TNumber> HardSigmoidBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the log sigmoid activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the log sigmoid function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the log sigmoid function.</returns>
    public ITensor<TNumber> LogSigmoid<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the log sigmoid function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the log sigmoid derivative function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the log sigmoid derivative function.</returns>
    public ITensor<TNumber> LogSigmoidBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Computes the GELU activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the GELU function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the GELU function.</returns>
    public ITensor<TNumber> GELU<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the GELU function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the GELU derivative function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the GELU derivative function.</returns>
    public ITensor<TNumber> GELUBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the softplus activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the softplus function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the softplus function.</returns>
    public ITensor<TNumber> SoftPlus<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the softplus function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the softplus derivative function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the softplus derivative function.</returns>
    public ITensor<TNumber> SoftPlusBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>;

    /// <summary>
    /// Computes the softsign activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the softsign function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the softsign function.</returns>
    public ITensor<TNumber> SoftSign<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the 1st derivative of the softsign function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the softsign derivative function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the softsign derivative function.</returns>
    public ITensor<TNumber> SoftSignBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;
}