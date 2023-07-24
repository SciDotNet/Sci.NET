// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Profiling;
using Sci.NET.MachineLearning.NeuralNetworks.Layers;
using Sci.NET.MachineLearning.NeuralNetworks.Parameters;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.NeuralNetworks;

/// <summary>
/// A neural network.
/// </summary>
/// <typeparam name="TNumber">The number type of the network.</typeparam>
[PublicAPI]
public class Network<TNumber> : IDisposable
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly List<ILayer<TNumber>> _layers;
    private ITensor<TNumber> _input;
    private ITensor<TNumber> _output;

    /// <summary>
    /// Initializes a new instance of the <see cref="Network{TNumber}"/> class.
    /// </summary>
    public Network()
    {
        _layers = new List<ILayer<TNumber>>();
        _input = Tensor.Zeros<TNumber>(1, 1);
        _output = Tensor.Zeros<TNumber>(1, 1);

        Profiler.LogGeneric("Network created.");
    }

    /// <summary>
    /// Finalizes an instance of the <see cref="Network{TNumber}"/> class.
    /// </summary>
    ~Network()
    {
        Dispose(false);
    }

    /// <summary>
    /// Adds a layer to the <see cref="Network{TNumber}"/>.
    /// </summary>
    /// <param name="layer">The layer to add.</param>
    public void AddLayer(ILayer<TNumber> layer)
    {
        _layers.Add(layer);

        Profiler.LogGeneric("Layer added");
    }

    /// <summary>
    /// Propagates the input through the <see cref="Network{TNumber}"/>.
    /// </summary>
    /// <param name="input">The input to propagate.</param>
    /// <returns>The output of the <see cref="Network{TNumber}"/>.</returns>
    public ITensor<TNumber> Forward(ITensor<TNumber> input)
    {
        _input = input;
        var result = input;

        foreach (var layer in _layers)
        {
            result = layer.Forward(result);
        }

        _output = result;
        return _output;
    }

    /// <summary>
    /// Propagates the error through the <see cref="Network{TNumber}"/>.
    /// </summary>
    /// <param name="error">The error to propagate.</param>
    public void Backward(ITensor<TNumber> error)
    {
        var result = error;

        foreach (var layer in _layers.Reverse<ILayer<TNumber>>())
        {
            result = layer.Backward(result);
        }
    }

    /// <summary>
    /// Gets the parameters of the <see cref="Network{TNumber}"/>.
    /// </summary>
    /// <returns>The parameters of the <see cref="Network{TNumber}"/>.</returns>
    public ParameterCollection<TNumber> Parameters()
    {
        var collection = new ParameterCollection<TNumber>();

        foreach (var layer in _layers)
        {
            collection.Add(layer.Parameters);
        }

        return collection;
    }

    /// <inheritdoc />
    public override int GetHashCode()
    {
        return HashCode.Combine(_layers, this);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes of the <see cref="Network{TNumber}"/>.
    /// </summary>
    /// <param name="disposing">Whether or not the object is being disposed.</param>
    protected virtual void Dispose(bool disposing)
    {
        Profiler.LogGeneric("Network disposed.");

        if (disposing)
        {
            foreach (var layer in _layers)
            {
                layer.Dispose();
            }

            _output.Dispose();
            _input.Dispose();
        }
    }
}