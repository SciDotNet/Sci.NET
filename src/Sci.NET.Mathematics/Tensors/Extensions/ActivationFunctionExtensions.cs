// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;

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
    public static ITensor<TNumber> Sigmoid<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        return TensorServiceProvider.GetTensorOperationServiceProvider()
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
    public static ITensor<TNumber> SigmoidPrime<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .SigmoidPrime(value);
    }

    /// <summary>
    /// Applies the ReLU function to the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The <see cref="ITensor{TNumber}"/> to apply the ReLU function to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the ReLU function.</returns>
    [DebuggerStepThrough]
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
    public static ITensor<TNumber> ReLUPrime<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .ReLUPrime(value);
    }

    /// <summary>
    /// Applies the softmax function to the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The <see cref="ITensor{TNumber}"/> to apply the softmax function to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the softmax function.</returns>
    [DebuggerStepThrough]
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
    public static ITensor<TNumber> SoftmaxPrime<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetActivationFunctionService()
            .SoftmaxPrime(value);
    }
}