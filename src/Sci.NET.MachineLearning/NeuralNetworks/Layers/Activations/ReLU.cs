// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.MachineLearning.NeuralNetworks.Parameters;
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
    /// <param name="device">The device to store the <see cref="ITensor{TNumber}"/> data on.</param>
    public ReLU(IDevice? device = null)
    {
        Device = device ?? Tensor.DefaultBackend.Device;
        Input = Tensor.Zeros<TNumber>(new Shape(1, 1), Device);
        Output = Tensor.Zeros<TNumber>(new Shape(1, 1), Device);
        Parameters = new ParameterSet<TNumber>(Device);
    }

    /// <inheritdoc />
    public bool IsDisposed { get; protected set; }

    /// <inheritdoc />
    public ITensor<TNumber> Input { get; private set; }

    /// <inheritdoc />
    public ITensor<TNumber> Output { get; private set; }

    /// <inheritdoc />
    public ParameterSet<TNumber> Parameters { get; }

    /// <inheritdoc />
    public IDevice Device { get; }

    /// <inheritdoc />
    public ITensor<TNumber> Forward(ITensor<TNumber> input)
    {
        Input.Dispose();
        Output.Dispose();
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
        To(new TDevice());
    }

    /// <inheritdoc />
    public void To(IDevice device)
    {
        Input.To(device);
        Output.To(device);
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