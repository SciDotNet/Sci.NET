// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.MachineLearning.NeuralNetworks.Parameters;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.NeuralNetworks.Optimizers;

/// <summary>
/// A gradient descent optimizer.
/// </summary>
/// <typeparam name="TNumber">The number type of the optimizer.</typeparam>
[PublicAPI]
public class GradientDescent<TNumber> : IOptimizer<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="GradientDescent{TNumber}"/> class.
    /// </summary>
    /// <param name="parameterSet">The parameters to optimize.</param>
    /// <param name="learningRate">The learning rate for the optimizer.</param>
    /// <param name="device">The device to store the <see cref="ITensor{TNumber}"/> data on.</param>
    public GradientDescent(
        ParameterCollection<TNumber> parameterSet,
        TNumber learningRate,
        IDevice? device = null)
    {
        Parameters = parameterSet;
        Device = device ?? new CpuComputeDevice();
        LearningRate = new Scalar<TNumber>(learningRate, Device.GetTensorBackend());
    }

    /// <inheritdoc />
    public ParameterCollection<TNumber> Parameters { get; }

    /// <inheritdoc />
    public Scalar<TNumber> LearningRate { get; }

    /// <inheritdoc />
    public IDevice Device { get; }

    /// <inheritdoc />
    public void Step()
    {
        _ = Parallel.ForEach(
            Parameters.GetAll(),
            namedParameter =>
            {
                using var gradient = namedParameter.Gradient;
                using var negativeLearningRate = LearningRate.Negate();
                using var delta = gradient.Multiply(negativeLearningRate);

                namedParameter.UpdateValue(delta);
            });
    }

    /// <inheritdoc />
    public void To<TDevice>()
        where TDevice : IDevice, new()
    {
        To(new TDevice());
    }

    /// <inheritdoc />
    public void To(IDevice device)
    {
        LearningRate.To(device);

        foreach (var namedParameter in Parameters.GetAll())
        {
            namedParameter.To(device);
        }
    }
}