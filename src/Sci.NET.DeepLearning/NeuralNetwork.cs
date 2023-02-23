// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.DeepLearning.Layers;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.DeepLearning;

/// <summary>
/// An implementation of a sequential neural network.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> stored within the network.</typeparam>
[PublicAPI]
public class NeuralNetwork<TNumber> : ISequentialNetwork<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly List<ILayer<TNumber>> _layers;

    /// <summary>
    /// Initializes a new instance of the <see cref="NeuralNetwork{TNumber}"/> class.
    /// </summary>
    public NeuralNetwork()
    {
        _layers = new List<ILayer<TNumber>>();
    }

    /// <inheritdoc />
    public IEnumerable<ILayer<TNumber>> Layers => _layers;

    /// <inheritdoc />
    public void AddLayer<TLayer>(TLayer layer)
        where TLayer : ILayer<TNumber>
    {
        _layers.Add(layer);
    }

    /// <summary>
    /// Propagates the <paramref name="input"/> through the network.
    /// </summary>
    /// <param name="input">The input to the network.</param>
    /// <returns>The result of the forward propagation.</returns>
    public ITensor<TNumber> Forward(ITensor<TNumber> input)
    {
        var output = input;

        foreach (var layer in _layers)
        {
            output = layer.Forward(output);
        }

        return output;
    }
}