// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.DeepLearning.Activations;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Mathematics.Tensors.LinearAlgebra;

namespace Sci.NET.DeepLearning.Layers;

/// <summary>
/// A fully connected layer.
/// </summary>
/// <typeparam name="TNumber">The type of number stored within the network.</typeparam>
[PublicAPI]
public class Dense<TNumber> : ILayer<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Dense{TNumber}"/> class.
    /// </summary>
    /// <param name="inputNeurons">The number of inputs to the <see cref="Dense{TNumber}"/> layer.</param>
    /// <param name="outputNeurons">The number of output units for the <see cref="Dense{TNumber}"/> layer.</param>
    public Dense(int inputNeurons, int outputNeurons)
    {
        Weights = Tensor.Random.Uniform(
            new Shape(inputNeurons, outputNeurons),
            TNumber.CreateChecked(0.0001),
            TNumber.CreateChecked(0.01),
            0);

        Biases = Tensor.Random.Uniform(
            new Shape(),
            TNumber.CreateChecked(0.0001),
            TNumber.CreateChecked(0.01),
            0);

        Output = Tensor.Zeros<TNumber>(new Shape(inputNeurons, outputNeurons));
    }

    /// <inheritdoc />
    public ITensor<TNumber> Weights { get; }

    /// <inheritdoc />
    public ITensor<TNumber> Biases { get; }

    /// <inheritdoc />
    public ITensor<TNumber> Output { get; set; }

    /// <summary>
    /// Gets or sets the activation function for the <see cref="Dense{TNumber}"/> layer.
    /// </summary>
    public IActivationFunction<TNumber> ActivationFunction { get; set; } = new EmptyActivationFunction<TNumber>();

    /// <inheritdoc />
    public ITensor<TNumber> Forward(ITensor<TNumber> input)
    {
        Output = Weights.Dot(input);
        Output += Biases;
        Output = ActivationFunction.Forward(Output);
        return Output;
    }
}