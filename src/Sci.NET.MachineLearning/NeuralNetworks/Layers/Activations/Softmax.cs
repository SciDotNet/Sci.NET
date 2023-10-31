// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.MachineLearning.NeuralNetworks.Parameters;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.NeuralNetworks.Layers.Activations;

/// <summary>
/// A softmax activation function layer.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
[PublicAPI]
public class Softmax<TNumber> : ILayer<TNumber>
    where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Softmax{TNumber}"/> class.
    /// </summary>
    /// <param name="device">The device to store the <see cref="ITensor{TNumber}"/> data on.</param>
    public Softmax(IDevice? device = null)
    {
        Device = device ?? Tensor.DefaultBackend.Device;
        Input = new Tensor<TNumber>(new Shape(1, 1), Device.GetTensorBackend());
        Output = new Tensor<TNumber>(new Shape(1, 1), Device.GetTensorBackend());
        Parameters = new ParameterSet<TNumber>(Device);
    }

    /// <inheritdoc />
    public IDevice Device { get; }

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
        Input.Dispose();
        Output.Dispose();

        Input = input;
        Output = Input.Softmax();

        return Output;
    }

    /// <inheritdoc />
    public ITensor<TNumber> Backward(ITensor<TNumber> error)
    {
        return error.SoftmaxPrime();
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
    /// Disposes of the managed resources used by the <see cref="Softmax{TNumber}"/>.
    /// </summary>
    /// <param name="disposing">Whether or not to dispose of managed resources.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            Input.Dispose();
            Output.Dispose();
        }
    }
}