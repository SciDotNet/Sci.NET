// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.MachineLearning.NeuralNetworks.Parameters;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.NeuralNetworks.Optimizers;

/// <summary>
/// An interface for an optimizer.
/// </summary>
/// <typeparam name="TNumber">The number type of the optimizer.</typeparam>
[PublicAPI]
public interface IOptimizer<TNumber> : ITensorLocalityOperations
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Gets the network of the optimizer.
    /// </summary>
    public ParameterCollection<TNumber> Parameters { get; }

    /// <summary>
    /// Gets the learning rate of the optimizer.
    /// </summary>
    public Scalar<TNumber> LearningRate { get; }

    /// <summary>
    /// Updates the weights and biases of the network.
    /// </summary>
    public void Step();
}