// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.DeepLearning.Activations;

/// <summary>
/// The empty activation function.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
[PublicAPI]
public class EmptyActivationFunction<TNumber> : IActivationFunction<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <inheritdoc />
    public ITensor<TNumber> Weights { get; } = new Tensor<TNumber>(new Shape());

    /// <inheritdoc />
    public ITensor<TNumber> Biases { get; } = new Tensor<TNumber>(new Shape());

    /// <inheritdoc />
    public ITensor<TNumber> Output { get; private set; } = new Tensor<TNumber>(new Shape());

    /// <inheritdoc />
    public ITensor<TNumber> Forward(ITensor<TNumber> input)
    {
        Output = input;
        return Output;
    }

    /// <inheritdoc />
    public ITensor<TNumber> Backward(ITensor<TNumber> error)
    {
        return error;
    }
}