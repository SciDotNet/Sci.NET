﻿// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.MachineLearning.NeuralNetworks.Parameters;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.NeuralNetworks.Layers;

/// <summary>
/// Flattens the input tensor.
/// </summary>
/// <typeparam name="TNumber">The number type of the layer.</typeparam>
[PublicAPI]
public class Flatten<TNumber> : ILayer<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Flatten{TNumber}"/> class.
    /// </summary>
    /// <param name="flattenAxis">The axis to flatten.</param>
    public Flatten(int flattenAxis = 1)
    {
        FlattenAxis = flattenAxis;
        Input = Tensor.Zeros<TNumber>(1, 1);
        Output = Tensor.Zeros<TNumber>(1, 1);
        Parameters = new ParameterSet<TNumber>();
    }

    /// <inheritdoc />
    public bool IsDisposed { get; }

    /// <inheritdoc />
    public ITensor<TNumber> Input { get; private set; }

    /// <inheritdoc />
    public ITensor<TNumber> Output { get; private set; }

    /// <inheritdoc />
    public ParameterSet<TNumber> Parameters { get; }

    /// <summary>
    /// Gets the axis to flatten.
    /// </summary>
    public int FlattenAxis { get; }

    /// <inheritdoc />
    public ITensor<TNumber> Forward(ITensor<TNumber> input)
    {
        Input = input;
        var inputShape = input.Shape.ToArray();
        var outputShape = new int[FlattenAxis + 1];

        for (var i = 0; i < FlattenAxis; i++)
        {
            outputShape[i] = inputShape[i];
        }

        outputShape[FlattenAxis] = inputShape[FlattenAxis..].Aggregate((a, b) => a * b);

        Output = Input.Reshape(outputShape);

        return Output;
    }

    /// <inheritdoc />
    public ITensor<TNumber> Backward(ITensor<TNumber> error)
    {
        return error;
    }

    /// <inheritdoc />
    public void To<TDevice>()
        where TDevice : IDevice, new()
    {
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
    /// Disposes of the <see cref="Flatten{TNumber}"/> layer.
    /// </summary>
    /// <param name="disposing">Whether or not the object is being disposed.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            Input.Dispose();
            Output.Dispose();
        }
    }
}