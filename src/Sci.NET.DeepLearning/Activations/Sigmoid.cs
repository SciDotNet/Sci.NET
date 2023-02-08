// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Mathematics.Tensors.Elementwise;

namespace Sci.NET.DeepLearning.Activations;

/// <summary>
/// The Rectified Linear Unit (ReLU) activation function.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="Sigmoid{TNumber}"/> activation function.</typeparam>
[PublicAPI]
public class Sigmoid<TNumber> : IActivationFunction<TNumber>
    where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
{
    /// <inheritdoc />
    public ITensor<TNumber> Weights { get; } = new Tensor<TNumber>(new Shape());

    /// <inheritdoc />
    public ITensor<TNumber> Biases { get; } = new Tensor<TNumber>(new Shape());

    /// <inheritdoc />
    public ITensor<TNumber> Output { get; } = new Tensor<TNumber>(new Shape());

    /// <inheritdoc />
    public ITensor<TNumber> Forward(ITensor<TNumber> input)
    {
        return TNumber.One / (TNumber.One + input.Exp());
    }
}