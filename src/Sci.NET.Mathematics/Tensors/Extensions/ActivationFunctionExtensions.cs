// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;
using Sci.NET.Common.Attributes;

#pragma warning disable IDE0130

// ReSharper disable once CheckNamespace
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// Extension methods for activation functions.
/// </summary>
[PublicAPI]
public static class ActivationFunctionExtensions
{
    /// <summary>
    /// Applies the sigmoid function to the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The <see cref="ITensor{TNumber}"/> to apply the sigmoid function to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the sigmoid function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\sigma(x) = \frac{1}{1 + e^{-x}}")]
    public static ITensor<TNumber> Sigmoid<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .Sigmoid(value);
    }

    /// <summary>
    /// Applies the derivative of the sigmoid function to the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The <see cref="ITensor{TNumber}"/> to apply the sigmoid derivative function to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the sigmoid derivative function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))")]
    public static ITensor<TNumber> SigmoidBackward<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .SigmoidBackward(value);
    }

    /// <summary>
    /// Applies the ReLU function to the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The <see cref="ITensor{TNumber}"/> to apply the ReLU function to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the ReLU function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{ReLU}\left(x\right)=\begin{cases}0&\text{if }x<0\\x&\text{if }x\geq0\end{cases}")]
    public static ITensor<TNumber> ReLU<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .ReLU(value);
    }

    /// <summary>
    /// Applies the derivative of the ReLU function to the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The <see cref="ITensor{TNumber}"/> to apply the ReLU derivative function to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the ReLU derivative function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{ReLU}'(x) = \begin{cases} 0 & \text{if } x < 0 \\ 1 & \text{if } x \geq 0 \end{cases}")]
    public static ITensor<TNumber> ReLUBackward<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .ReLUBackward(value);
    }

    /// <summary>
    /// Applies the softmax function to the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The <see cref="ITensor{TNumber}"/> to apply the softmax function to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the softmax function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}")]
    public static ITensor<TNumber> Softmax<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .Softmax(value);
    }

    /// <summary>
    /// Applies the derivative of the softmax function to the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The <see cref="ITensor{TNumber}"/> to apply the softmax derivative function to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the softmax derivative function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{softmax}'(x_i) = \text{softmax}(x_i) \cdot (1 - \text{softmax}(x_i))")]
    public static ITensor<TNumber> SoftmaxBackward<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .SoftmaxBackward(value);
    }

    /// <summary>
    /// Computes the Leaky ReLU activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Leaky ReLU function on.</param>
    /// <param name="alpha">The alpha value for the Leaky ReLU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the Leaky ReLU function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha \cdot x & \text{if } x < 0 \end{cases}")]
    public static ITensor<TNumber> LeakyReLU<TNumber>(this ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .LeakyReLU(value, alpha);
    }

    /// <summary>
    /// Computes the 1st derivative of the Leaky ReLU function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Leaky ReLU derivative function on.</param>
    /// <param name="alpha">The alpha value for the Leaky ReLU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the Leaky ReLU derivative function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{LeakyReLU}'(x) = \begin{cases} 1 & \text{if } x \geq 0 \\ \alpha & \text{if } x < 0 \end{cases}")]
    public static ITensor<TNumber> LeakyReLUBackward<TNumber>(this ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .LeakyReLUBackward(value, alpha);
    }

    /// <summary>
    /// Computes the ELU activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the ELU function on.</param>
    /// <param name="alpha">The alpha value for the ELU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the ELU function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{ELU}(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha \cdot (e^x - 1) & \text{if } x < 0 \end{cases}")]
    public static ITensor<TNumber> Elu<TNumber>(this ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .Elu(value, alpha);
    }

    /// <summary>
    /// Computes the 1st derivative of the ELU function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the ELU derivative function on.</param>
    /// <param name="alpha">The alpha value for the ELU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the ELU derivative function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{ELU}'(x) = \begin{cases} 1 & \text{if } x \geq 0 \\ \alpha \cdot e^x & \text{if } x < 0 \end{cases}")]
    public static ITensor<TNumber> EluBackward<TNumber>(this ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .EluBackward(value, alpha);
    }

    /// <summary>
    /// Computes the CELU activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the CELU function on.</param>
    /// <param name="alpha">The alpha value for the CELU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the CELU function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{CELU}(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha \cdot (e^{\frac{x}{\alpha}} - 1) & \text{if } x < 0 \end{cases}")]
    public static ITensor<TNumber> Celu<TNumber>(this ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .Celu(value, alpha);
    }

    /// <summary>
    /// Computes the 1st derivative of the CELU function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the CELU derivative function on.</param>
    /// <param name="alpha">The alpha value for the CELU function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the CELU derivative function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{CELU}'(x) = \begin{cases} 1 & \text{if } x \geq 0 \\ \frac{e^{\frac{x}{\alpha}}}{\alpha + e^{\frac{x}{\alpha}}} & \text{if } x < 0 \end{cases}")]
    public static ITensor<TNumber> CeluBackward<TNumber>(this ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .CeluBackward(value, alpha);
    }

    /// <summary>
    /// Computes the Swish activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Swish function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the Swish function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{Swish}(x) = \frac{x}{1 + e^{-x}}")]
    public static ITensor<TNumber> Swish<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .Swish(value);
    }

    /// <summary>
    /// Computes the 1st derivative of the Swish function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Swish derivative function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the Swish derivative function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{Swish}'(x) = \text{Swish}(x) + \frac{1}{1 + e^{-x}} \cdot (1 - \text{Swish}(x))")]
    public static ITensor<TNumber> SwishBackward<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .SwishBackward(value);
    }

    /// <summary>
    /// Computes the Mish activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Mish function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the Mish function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"mish\left(x\right)=x\tanh\left(\ln\left(1+e^{x}\right)\right)")]
    [MathematicExpression(1, @"mish\left(x\right)=x\cdot\frac{e^{\ln\left(e^{x}+1\right)}-e^{-\ln\left(e^{x}+1\right)}}{e^{\ln\left(e^{x}+1\right)}+e^{-\ln\left(e^{x}+1\right)}}")]
    public static ITensor<TNumber> Mish<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IHyperbolicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .Mish(value);
    }

    /// <summary>
    /// Computes the 1st derivative of the Mish function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the Mish derivative function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the Mish derivative function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"mish'\left(x\right)=\tanh\left(\ln\left(1+e^{x}\right)\right)+\frac{xe^{x}\operatorname{sech}^{2}\left(\ln\left(1+e^{x}\right)\right)}{e^{x}+1}")]
    [MathematicExpression(1, @"mish'\left(x\right)=\frac{-1+\left(1+e^{x}\right)^{2}}{1+\left(1+e^{x}\right)^{2}}-\frac{2e^{x}\left(1+e^{x}\right)\left(-1+\left(1+e^{x}\right)^{2}\right)x}{\left(1+\left(1+e^{x}\right)^{2}\right)^{2}}+\frac{2e^{x}\left(1+e^{x}\right)x}{1+\left(1+e^{x}\right)^{2}}")]
    public static ITensor<TNumber> MishBackward<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IHyperbolicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .MishBackward(value);
    }

    /// <summary>
    /// Computes the hard Tanh activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the hard Tanh function on.</param>
    /// <param name="min">The minimum value of the hard Tanh function.</param>
    /// <param name="max">The maximum value of the hard Tanh function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the hard Tanh function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{HardTanh}(x) = \begin{cases} \text{min} & \text{if } x < \text{min} \\ x & \text{if } \text{min} \leq x \leq \text{max} \\ \text{max} & \text{if } x > \text{max} \end{cases}")]
    public static ITensor<TNumber> HardTanh<TNumber>(this ITensor<TNumber> value, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .HardTanh(value, min, max);
    }

    /// <summary>
    /// Computes the 1st derivative of the hard Tanh function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the hard Tanh derivative function on.</param>
    /// <param name="min">The minimum value of the hard Tanh function.</param>
    /// <param name="max">The maximum value of the hard Tanh function.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the hard Tanh derivative function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{HardTanh}'(x) = \begin{cases} 0 & \text{if } x < \text{min} \\ 1 & \text{if } \text{min} \leq x \leq \text{max} \\ 0 & \text{if } x > \text{max} \end{cases}")]
    public static ITensor<TNumber> HardTanhBackward<TNumber>(this ITensor<TNumber> value, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .HardTanhBackward(value, min, max);
    }

    /// <summary>
    /// Computes the hard sigmoid activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the hard sigmoid function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the hard sigmoid function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{HardSigmoid}(x) = \begin{cases} 0 & \text{if } x < -1 \\ 1 & \text{if } x > 1 \\ 0.5x + 0.5 & \text{if } -1 \leq x \leq 1 \end{cases}")]
    public static ITensor<TNumber> HardSigmoid<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .HardSigmoid(value);
    }

    /// <summary>
    /// Computes the 1st derivative of the hard sigmoid function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the hard sigmoid derivative function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the hard sigmoid derivative function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{HardSigmoid}'(x) = \begin{cases} 0 & \text{if } x < -1 \\ 0 & \text{if } x > 1 \\ 0.5 & \text{if } -1 \leq x \leq 1 \end{cases}")]
    public static ITensor<TNumber> HardSigmoidBackward<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .HardSigmoidBackward(value);
    }

    /// <summary>
    /// Computes the log sigmoid activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the log sigmoid function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the log sigmoid function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{LogSigmoid}(x) = \ln\left(\frac{1}{1 + e^{-x}}\right)")]
    public static ITensor<TNumber> LogSigmoid<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .LogSigmoid(value);
    }

    /// <summary>
    /// Computes the 1st derivative of the log sigmoid function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the log sigmoid derivative function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the log sigmoid derivative function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{LogSigmoid}'(x) = \frac{1}{1 + e^{x}}")]
    public static ITensor<TNumber> LogSigmoidBackward<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .LogSigmoidBackward(value);
    }

    /// <summary>
    /// Computes the GELU activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the GELU function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the GELU function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"GELU\left(x\right)=0.5x\left(1+\frac{2}{\sqrt{\pi}}\int_{0}^{\frac{x}{\sqrt{2}}}e^{-t^{2}}dt\right)")]
    [MathematicExpression(1, @"GELU\left(x\right)\approx0.5x\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}\left(x+0.044715x^{3}\right)\right)\right)")]
    public static ITensor<TNumber> GELU<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .GELU(value);
    }

    /// <summary>
    /// Computes the 1st derivative of the GELU function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the GELU derivative function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the GELU derivative function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"GELU'\left(x\right)=\frac{0.3989422804014327x}{e^{\frac{x^{2}}{2}}}+0.5\left(1+\frac{2}{\sqrt{\pi}}\int_{0}^{\frac{x}{\sqrt{2}}}e^{-t^{2}}dt\right)")]
    [MathematicExpression(1, @"GELU'\left(x\right)=0.3989422804014327x\left(1+0.134145x^{2}\right)\operatorname{sech}^{2}\left(\sqrt{\frac{2}{\pi}}\left(x+0.044715x^{3}\right)\right)+0.5\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}\left(x+0.044715x^{3}\right)\right)\right)")]
    public static ITensor<TNumber> GELUBackward<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .GELUBackward(value);
    }

    /// <summary>
    /// Computes the softplus activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the softplus function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the softplus function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{SoftPlus}(x) = \ln\left(1 + e^{x}\right)")]
    public static ITensor<TNumber> SoftPlus<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .SoftPlus(value);
    }

    /// <summary>
    /// Computes the 1st derivative of the softplus function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the softplus derivative function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the softplus derivative function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{SoftPlus}'(x) = \frac{e^{x}}{1 + e^{x}}")]
    public static ITensor<TNumber> SoftPlusBackward<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .SoftPlusBackward(value);
    }

    /// <summary>
    /// Computes the softsign activation function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the softsign function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the softsign function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{SoftSign}(x) = \frac{x}{1 + |x|}")]
    public static ITensor<TNumber> SoftSign<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .SoftSign(value);
    }

    /// <summary>
    /// Computes the 1st derivative of the softsign function on the given <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The value to compute the softsign derivative function on.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the softsign derivative function.</returns>
    [DebuggerStepThrough]
    [MathematicExpression(0, @"\text{SoftSign}'(x) = \frac{1}{(1 + |x|)^2}")]
    public static ITensor<TNumber> SoftSignBackward<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .SoftSignBackward(value);
    }
}