// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.NeuralNetworks.Layers.Activations;

/// <summary>
/// Rectified Linear Unit activation function.
/// </summary>
/// <typeparam name="TNumber">The number type of the layer.</typeparam>
[PublicAPI]
public class ReLU<TNumber> : ILayer<TNumber>
    where TNumber : unmanaged, IFloatingPoint<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ReLU{TNumber}"/> class.
    /// </summary>
    public ReLU()
    {
        Input = Tensor.Zeros<TNumber>(1, 1);
        Output = Tensor.Zeros<TNumber>(1, 1);
    }

    /// <inheritdoc />
    public bool IsDisposed { get; protected set; }

    /// <inheritdoc />
    public ITensor<TNumber> Input { get; private set; }

    /// <inheritdoc />
    public ITensor<TNumber> Output { get; private set; }

    /// <inheritdoc />
    public ITensor<TNumber> Forward(ITensor<TNumber> input)
    {
        Input = input;
        Output = Input.ReLU();

        return Output;
    }

    /// <inheritdoc />
    public ITensor<TNumber> Backward(ITensor<TNumber> error)
    {
        return error.ReLUPrime();
    }

    /// <inheritdoc />
    public void To<TDevice>()
        where TDevice : IDevice, new()
    {
        Input = Input.To<TDevice>();
        Output = Output.To<TDevice>();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes of the <see cref="ReLU{TNumber}"/> layer.
    /// </summary>
    /// <param name="disposing">Whether or not the object is being disposed.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
        }

        IsDisposed = true;
    }
}