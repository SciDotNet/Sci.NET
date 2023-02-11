// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.DeepLearning.Layers;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.DeepLearning;

/// <summary>
/// An interface for sequential neural networks.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> used in the network.</typeparam>
[PublicAPI]
public interface ISequentialNetwork<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Gets the layers of the network.
    /// </summary>
    public IEnumerable<ILayer<TNumber>> Layers { get; }

    /// <summary>
    /// Adds a layer to the network.
    /// </summary>
    /// <param name="layer">The instance of the layer to add to the network.</param>
    /// <typeparam name="TLayer">The type of layer to add to the network.</typeparam>
    public void AddLayer<TLayer>(TLayer layer)
        where TLayer : ILayer<TNumber>;
}