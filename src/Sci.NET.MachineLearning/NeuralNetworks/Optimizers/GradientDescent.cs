// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.MachineLearning.NeuralNetworks.Parameters;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.NeuralNetworks.Optimizers;

/// <summary>
/// A gradient descent optimizer.
/// </summary>
/// <typeparam name="TNumber">The number type of the optimizer.</typeparam>
[PublicAPI]
public class GradientDescent<TNumber> : ITensorLocalityOperations, IOptimizer<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="GradientDescent{TNumber}"/> class.
    /// </summary>
    /// <param name="parameterSet">The parameters to optimize.</param>
    /// <param name="learningRate">The learning rate for the optimizer.</param>
    public GradientDescent(
        ParameterCollection<TNumber> parameterSet,
        TNumber learningRate)
    {
        Parameters = parameterSet;
        LearningRate = new Scalar<TNumber>(learningRate);
    }

    /// <inheritdoc />
    public ParameterCollection<TNumber> Parameters { get; }

    /// <summary>
    /// Gets the learning rate.
    /// </summary>
    public Scalar<TNumber> LearningRate { get; }

    /// <inheritdoc />
    public void Step()
    {
        foreach (var parameterSets in Parameters)
        {
            foreach (var namedParameter in parameterSets)
            {
                var gradient = namedParameter.Gradient;
                var delta = gradient * LearningRate.Negate();

                namedParameter.UpdateValue(delta);
            }
        }
    }

    /// <inheritdoc />
    public void To<TDevice>()
        where TDevice : IDevice, new()
    {
        LearningRate.To<TDevice>();
    }
}