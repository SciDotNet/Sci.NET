// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.MachineLearning.NeuralNetworks.Parameters;
using Sci.NET.Mathematics.Backends.Devices;
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
    private const string WeightsParameterName = "weights";
    private const string BiasesParameterName = "biases";

    /// <summary>
    /// Initializes a new instance of the <see cref="Dense{TNumber}"/> class.
    /// </summary>
    /// <param name="inputFeatures">The number of input features.</param>
    /// <param name="outputFeatures">The number of output features.</param>
    /// <param name="device">The device to store the <see cref="ITensor{TNumber}"/> data on.</param>
    public Dense(int inputFeatures, int outputFeatures, IDevice? device = null)
    {
        InputFeatures = inputFeatures;
        OutputFeatures = outputFeatures;
        Device = device ?? Tensor.DefaultBackend.Device;

        Parameters = new ParameterSet<TNumber>(Device)
        {
            { WeightsParameterName, new Shape(inputFeatures, outputFeatures) },
            { BiasesParameterName, new Shape(outputFeatures) }
        };

        Input = Tensor.Zeros<TNumber>(1, 1);
        Output = Tensor.Zeros<TNumber>(1, 1);
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
    public IDevice Device { get; }

    /// <inheritdoc />
    public ITensor<TNumber> Input { get; private set; }

    /// <inheritdoc />
    public ITensor<TNumber> Output { get; private set; }

    /// <inheritdoc />
    public ParameterSet<TNumber> Parameters { get; }

    /// <inheritdoc />
    public ITensor<TNumber> Forward(ITensor<TNumber> input)
    {
        Input.Dispose();
        Output.Dispose();

#pragma warning disable SA1013
        if (input.Shape[1] != InputFeatures)
        {
            throw new InvalidShapeException(
                $"Input shape {input.Shape} does not match expected shape" +
                $" {new int[] { input.Shape[0], InputFeatures }}.");
        }
#pragma warning restore SA1013

        var weights = Parameters[WeightsParameterName].Value;
        var bias = Parameters[BiasesParameterName].Value;

        using var weightedInput = input.Dot(weights);
        var weightedInputPlusBias = weightedInput.Add(bias);

        Input = input;
        Output = weightedInputPlusBias;

        return Output;
    }

    /// <inheritdoc />
    public ITensor<TNumber> Backward(ITensor<TNumber> error)
    {
        using var m = new Scalar<TNumber>(TNumber.CreateChecked(error.Shape[0]));

        var weights = Parameters[WeightsParameterName].Value;

        Parameters[WeightsParameterName].SetGradient(Input.Transpose().Dot(error));
        Parameters[BiasesParameterName].SetGradient(error.Sum([0], keepDims: true));

        return weights.Dot(error.Transpose()).Transpose();
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
        Input.To(device);
        Output.To(device);
        Parameters.To(device);
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
        if (!disposing)
        {
            return;
        }

        IsDisposed = true;

        Parameters.Dispose();
        Input.Dispose();
        Output.Dispose();
    }
}