// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.MachineLearning.NeuralNetworks.Parameters.Initializers;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.NeuralNetworks.Parameters;

/// <summary>
/// A named parameter.
/// </summary>
/// <typeparam name="TNumber">The number type of the parameter.</typeparam>
[PublicAPI]
public class NamedParameter<TNumber> : ITensorLocalityOperations, IDisposable
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="NamedParameter{TNumber}"/> class.
    /// </summary>
    /// <param name="name">The name of the parameter.</param>
    /// <param name="shape">The shape of the parameter.</param>
    /// <param name="device">The device to store the <see cref="ITensor{TNumber}"/> data on.</param>
    public NamedParameter(string name, Shape shape, IDevice? device = null)
    {
        Name = name;
        Device = device ?? new CpuComputeDevice();
        Gradient = Initializer.Initialize<TNumber>(shape, Device);
        Value = Initializer.Initialize<TNumber>(shape, Device);
    }

    /// <summary>
    /// Gets the name of the parameter.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the value of the parameter.
    /// </summary>
    public ITensor<TNumber> Value { get; private set; }

    /// <summary>
    /// Gets the gradient of the parameter.
    /// </summary>
    public ITensor<TNumber> Gradient { get; private set; }

    /// <inheritdoc />
    public IDevice Device { get; }

    /// <summary>
    /// Gets the initializer for the parameter.
    /// </summary>
    public IParameterInitializer Initializer { get; init; } = new DefaultParameterInitializer();

    /// <inheritdoc />
    public void To<TDevice>()
        where TDevice : IDevice, new()
    {
        Gradient.To<TDevice>();
        Value.To<TDevice>();
    }

    /// <summary>
    /// Updates the value of the parameter.
    /// </summary>
    /// <param name="amount">The amount to change the parameter.</param>
    public void UpdateValue(ITensor<TNumber> amount)
    {
        Value.Dispose();
        Value = Value.Add(amount);
    }

    /// <summary>
    /// Sets the gradient of the parameter.
    /// </summary>
    /// <param name="gradient">The gradient to set.</param>
    public void SetGradient(ITensor<TNumber> gradient)
    {
        Gradient.Dispose();
        Gradient = gradient;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes of the <see cref="NamedParameter{TNumber}"/>.
    /// </summary>
    /// <param name="disposing">Whether the <see cref="NamedParameter{TNumber}"/> is disposing.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            Gradient.Dispose();
            Gradient.Dispose();
        }
    }
}