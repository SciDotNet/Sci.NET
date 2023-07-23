// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Profiling;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Backends.Managed;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.MachineLearning.NeuralNetworks.Layers;

/// <summary>
/// A dense neural network layer.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="Dense{TNumber}"/> layer.</typeparam>
[PublicAPI]
public class Dense<TNumber> : ILayer<TNumber>
    where TNumber : unmanaged, IFloatingPoint<TNumber>
{
    private ITensor<TNumber> _weights;
    private ITensor<TNumber> _biases;
    private ITensor<TNumber> _dw;
    private ITensor<TNumber> _db;

    /// <summary>
    /// Initializes a new instance of the <see cref="Dense{TNumber}"/> class.
    /// </summary>
    /// <param name="inputFeatures">The number of input features.</param>
    /// <param name="outputFeatures">The number of output features.</param>
    public Dense(int inputFeatures, int outputFeatures)
    {
        InputFeatures = inputFeatures;
        OutputFeatures = outputFeatures;

        _weights = new Matrix<TNumber>(inputFeatures, outputFeatures, ManagedTensorBackend.Instance);
        _biases = new Mathematics.Tensors.Vector<TNumber>(outputFeatures, ManagedTensorBackend.Instance);
        _dw = new Scalar<TNumber>();
        _db = new Scalar<TNumber>();

        Input = Tensor.Zeros<TNumber>(1, 1);
        Output = Tensor.Zeros<TNumber>(1, 1);

        for (var i = 0; i < _weights.Handle.Length; i++)
        {
#pragma warning disable CA5394
            _weights.Handle[i] = TNumber.CreateChecked(Random.Shared.NextDouble() / 100.0d);
#pragma warning restore CA5394
        }

        for (var i = 0; i < _weights.Handle.Length; i++)
        {
#pragma warning disable CA5394
            _weights.Handle[i] = TNumber.CreateChecked(Random.Shared.NextDouble() / 100.0d);
#pragma warning restore CA5394
        }
    }

    /// <summary>
    /// Gets the number of input features.
    /// </summary>
    public int InputFeatures { get; }

    /// <summary>
    /// Gets the number of output features.
    /// </summary>
    public int OutputFeatures { get; }

    /// <inheritdoc />
    public bool IsDisposed { get; private set; }

    /// <inheritdoc />
    public ITensor<TNumber> Input { get; private set; }

    /// <inheritdoc />
    public ITensor<TNumber> Output { get; private set; }

    /// <inheritdoc />
    public ITensor<TNumber> Forward(ITensor<TNumber> input)
    {
#pragma warning disable SA1013
        if (input.Shape[1] != InputFeatures)
        {
            throw new InvalidShapeException(
                $"Input shape {input.Shape} does not match expected" +
                $"shape {new int[] { input.Shape[0], InputFeatures }}.");
        }
#pragma warning restore SA1013

        var output = (input.Dot(_weights).Transpose() + _biases).Transpose();

        Input = input;
        Output = output;

        return output;
    }

    /// <inheritdoc />
    public ITensor<TNumber> Backward(ITensor<TNumber> error)
    {
        using var m = new Scalar<TNumber>(TNumber.CreateChecked(error.Shape[0]));
        _dw = Input.Transpose().Dot(error);
        _db = error.Sum(new int[] { 0 });

        return _weights.Transpose().Dot(error);
    }

    /// <inheritdoc />
    public void To<TDevice>()
        where TDevice : IDevice, new()
    {
        _weights = _weights.To<TDevice>().AsMatrix();
        _biases = _biases.To<TDevice>().AsVector();
        _dw = _dw.To<TDevice>().AsScalar();
        _db = _db.To<TDevice>().AsScalar();
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
    /// Disposes the layer.
    /// </summary>
    /// <param name="disposing">A value indicating whether the instance is disposing.</param>
    protected virtual void Dispose(bool disposing)
    {
        Profiler.LogObjectDisposed("Dense layer disposed", GetHashCode());

        if (!disposing)
        {
            return;
        }

        IsDisposed = true;

        _weights.Dispose();
        _biases.Dispose();
        _dw.Dispose();
        _db.Dispose();
        Input.Dispose();
        Output.Dispose();
    }
}