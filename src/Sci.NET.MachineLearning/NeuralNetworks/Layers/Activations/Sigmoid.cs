// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.MachineLearning.NeuralNetworks.Parameters;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.NeuralNetworks.Layers.Activations;

/// <summary>
/// Sigmoid activation function.
/// </summary>
/// <typeparam name="TNumber">The number type of the layer.</typeparam>
[PublicAPI]
public class Sigmoid<TNumber> : ILayer<TNumber>
    where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Sigmoid{TNumber}"/> class.
    /// </summary>
    /// <param name="device">The device to store the <see cref="ITensor{TNumber}"/> data on.</param>
    public Sigmoid(IDevice? device = null)
    {
        Device = device ?? new CpuComputeDevice();
        Parameters = new ParameterSet<TNumber>(Device);
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
    public ParameterSet<TNumber> Parameters { get; }

    /// <inheritdoc />
    public IDevice Device { get; }

    /// <inheritdoc />
    public ITensor<TNumber> Forward(ITensor<TNumber> input)
    {
        Input.Dispose();
        Output.Dispose();

        Input = input;
        Output = input.Sigmoid();

        return Output;
    }

    /// <inheritdoc />
    public ITensor<TNumber> Backward(ITensor<TNumber> error)
    {
        return error.SigmoidPrime();
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
    /// Disposes of the <see cref="Sigmoid{TNumber}"/> layer.
    /// </summary>
    /// <param name="disposing">Whether or not the object is being disposed.</param>
    protected virtual void Dispose(bool disposing)
    {
        IsDisposed = true;
    }
}