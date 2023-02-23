// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Mathematics.Tensors.Backends;

namespace Sci.NET.DeepLearning.Layers;

/// <summary>
/// A 2D convolutional layer.
/// </summary>
/// <typeparam name="TNumber">The type of number stored within the network.</typeparam>
[PublicAPI]
public class Conv2D<TNumber> : ILayer<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Conv2D{TNumber}"/> class.
    /// </summary>
    /// <param name="inputChannels">The number of input channels.</param>
    /// <param name="outputChannels">The number of output channels.</param>
    /// <param name="kernelWidth">The kernel width.</param>
    /// <param name="kernelHeight">The kernel height.</param>
    /// <param name="strideX">The stride in the x dimension.</param>
    /// <param name="strideY">The stride in the y dimension.</param>
    /// <param name="dilationX">The dilation in the x dimension.</param>
    /// <param name="dilationY">The dilation in the y dimension.</param>
    public Conv2D(
        int inputChannels,
        int outputChannels,
        int kernelWidth,
        int kernelHeight,
        int strideX,
        int strideY,
        int dilationX,
        int dilationY)
    {
        StrideX = strideX;
        StrideY = strideY;
        DilationX = dilationX;
        DilationY = dilationY;

        Weights = Tensor.Random.Uniform(
            new Shape(kernelWidth, kernelHeight, inputChannels, outputChannels),
            TNumber.CreateChecked(0.001),
            TNumber.CreateChecked(0.1),
            DateTime.Now.Ticks);

        Biases = new Tensor<TNumber>(new Shape());
        Output = new Tensor<TNumber>(new Shape(0));
    }

    /// <summary>
    /// Gets the stride in the x dimension.
    /// </summary>
    public int StrideX { get; }

    /// <summary>
    /// Gets the stride in the y dimension.
    /// </summary>
    public int StrideY { get; }

    /// <summary>
    /// Gets the dilation in the x dimension.
    /// </summary>
    public int DilationX { get; }

    /// <summary>
    /// Gets the dilation in the y dimension.
    /// </summary>
    public int DilationY { get; }

    /// <inheritdoc />
    public ITensor<TNumber> Weights { get; }

    /// <inheritdoc />
    public ITensor<TNumber> Biases { get; }

    /// <inheritdoc />
    public ITensor<TNumber> Output { get; private set; }

    /// <inheritdoc />
    public ITensor<TNumber> Forward(ITensor<TNumber> input)
    {
        Output.Dispose();

        Output = TensorBackend.Instance.NeuralNetwork.Conv2d(
            input,
            Weights,
            StrideX,
            StrideY,
            DilationX,
            DilationY);
        return Output;
    }
}