// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;
using Sci.NET.MachineLearning.NeuralNetworks.Parameters;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.MachineLearning.NeuralNetworks.Layers;

/// <summary>
/// A convolutional layer.
/// </summary>
/// <typeparam name="TNumber">The number type of the Conv2D operation.</typeparam>
[PublicAPI]
public class Conv2D<TNumber> : Conv2dParameters, ILayer<TNumber>
    where TNumber : unmanaged, IFloatingPoint<TNumber>
{
    private const string WeightsParameterName = "weights";
    private const string BiasesParameterName = "biases";

    /// <summary>
    /// Initializes a new instance of the <see cref="Conv2D{TNumber}"/> class.
    /// </summary>
    public Conv2D()
    {
        Input = Tensor.Zeros<TNumber>(1, 1);
        Output = Tensor.Zeros<TNumber>(1, 1);

        Parameters =
            new ParameterSet<TNumber>
            {
                { WeightsParameterName, new Shape(Filters, KernelSizeY, KernelSizeX) },
                { BiasesParameterName, new Shape(Filters) }
            };
    }

    /// <inheritdoc />
    public bool IsDisposed { get; }

    /// <inheritdoc />
    public ITensor<TNumber> Input { get; private set; }

    /// <inheritdoc />
    public ITensor<TNumber> Output { get; private set; }

    /// <inheritdoc />
    public ParameterSet<TNumber> Parameters { get; }

    /// <inheritdoc />
    public ITensor<TNumber> Forward(ITensor<TNumber> input)
    {
        if (input.Shape.Rank == 3)
        {
            input = input.Reshape(1, input.Shape[0], input.Shape[1], input.Shape[2]).AsTensor();
        }

        if (input.Shape.Rank != 4)
        {
            throw new InvalidShapeException($"The input for a {nameof(Conv2D<TNumber>)} must be a 3D or 4D tensor.");
        }

        var inputTensor = input.AsTensor();
        ref var weights = ref Parameters[WeightsParameterName].Value;

        Input = input;

        Output = inputTensor.Conv2DForward(
            weights.AsTensor(),
            StrideX,
            StrideY,
            PaddingX,
            PaddingY,
            DilationX,
            DilationY);

        return Output;
    }

    /// <inheritdoc />
    public ITensor<TNumber> Backward(ITensor<TNumber> error)
    {
        throw new UnreachableException();
    }

    /// <inheritdoc />
    public void To<TDevice>()
        where TDevice : IDevice, new()
    {
        Parameters.To<TDevice>();
        Input.To<TDevice>();
        Output.To<TDevice>();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes of the layer.
    /// </summary>
    /// <param name="disposing">A value indicating whether the instance is disposing.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            Input.Dispose();
            Output.Dispose();
            Parameters.Dispose();
        }
    }
}