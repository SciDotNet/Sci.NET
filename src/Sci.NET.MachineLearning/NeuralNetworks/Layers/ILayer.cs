// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.MachineLearning.NeuralNetworks.Parameters;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.NeuralNetworks.Layers;

/// <summary>
/// An interface for a neural network layer.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="ILayer{TNumber}"/>.</typeparam>
[PublicAPI]
public interface ILayer<TNumber> : ITensorLocalityOperations, IDisposable
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Gets a value indicating whether the layer has been disposed.
    /// </summary>
    public bool IsDisposed { get; }

    /// <summary>
    /// Gets the input of the layer.
    /// </summary>
    public ITensor<TNumber> Input { get; }

    /// <summary>
    /// Gets the output of the layer.
    /// </summary>
    public ITensor<TNumber> Output { get; }

    /// <summary>
    /// Gets the parameters of the layer.
    /// </summary>
    public ParameterSet<TNumber> Parameters { get; }

    /// <summary>
    /// Propagates the input through the layer.
    /// </summary>
    /// <param name="input">The input to the network.</param>
    /// <returns>The activation of the layer given <paramref name="input"/>.</returns>
    public ITensor<TNumber> Forward(ITensor<TNumber> input);

    /// <summary>
    /// Propagates the error through the layer.
    /// </summary>
    /// <param name="error">The gradient of the input.</param>
    /// <returns>The gradient of the current layer.</returns>
    public ITensor<TNumber> Backward(ITensor<TNumber> error);
}