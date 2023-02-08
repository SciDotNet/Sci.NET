// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.DeepLearning.Layers;

/// <summary>
/// A layer in a neural network.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> stored within the layer.</typeparam>
[PublicAPI]
public interface ILayer<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Gets the weights of the <see cref="ILayer{TNumber}"/>.
    /// </summary>
    public ITensor<TNumber> Weights { get; }

    /// <summary>
    /// Gets the biases of the <see cref="ILayer{TNumber}"/>.
    /// </summary>
    public ITensor<TNumber> Biases { get; }

    /// <summary>
    /// Gets the last output of the <see cref="ILayer{TNumber}"/>.
    /// </summary>
    public ITensor<TNumber> Output { get; }

    /// <summary>
    /// Propagates the input through the <see cref="ILayer{TNumber}"/>.
    /// </summary>
    /// <param name="input">The input to propagate through the <see cref="ILayer{TNumber}"/>.</param>
    /// <returns>The result of the <see cref="ILayer{TNumber}"/>.</returns>
    public ITensor<TNumber> Forward(ITensor<TNumber> input);
}