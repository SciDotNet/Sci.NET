// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.NeuralNetworks.Layers.Activations;

/// <summary>
/// A softmax activation function.
/// </summary>
/// <typeparam name="TNumber">The number type of the layer.</typeparam>
public class Softmax<TNumber> : ILayer<TNumber>
    where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Softmax{TNumber}"/> class.
    /// </summary>
    public Softmax()
    {
        Input = new Tensor<TNumber>();
        Output = new Tensor<TNumber>();
    }

    /// <inheritdoc />
    public bool IsDisposed { get; private set; }

    /// <inheritdoc />
    public ITensor<TNumber> Input { get; private set; }

    /// <inheritdoc />
    public ITensor<TNumber> Output { get; private set; }

    /// <inheritdoc />
    public ITensor<TNumber> Forward(ITensor<TNumber> input)
    {
        Input = input;
        Output = Input.Softmax();

        return Output;
    }

    /// <inheritdoc />
    public ITensor<TNumber> Backward(ITensor<TNumber> error)
    {
        throw new NotSupportedException();
    }

    /// <inheritdoc />
    public void To<TDevice>()
        where TDevice : IDevice, new()
    {
        Input = Input.To<TDevice>();
        Output = Output.To<TDevice>();
    }

    /// <summary>
    /// Creates a new softmax activation function.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Creates a new softmax activation function.
    /// </summary>
    /// <param name="disposing">Whether or not to dispose of managed resources.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            Input.Dispose();
            Output.Dispose();
            IsDisposed = true;
        }
    }
}