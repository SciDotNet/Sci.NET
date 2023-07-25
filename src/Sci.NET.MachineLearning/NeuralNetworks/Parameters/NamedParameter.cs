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
    private ITensor<TNumber> _value;
    private ITensor<TNumber> _gradient;

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
        _gradient = Initializer.Initialize<TNumber>(shape, Device);
        _value = Initializer.Initialize<TNumber>(shape, Device);
    }

    /// <summary>
    /// Gets the name of the parameter.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the value of the parameter.
    /// </summary>
    public ref ITensor<TNumber> Value => ref _value;

    /// <summary>
    /// Gets the gradient of the parameter.
    /// </summary>
    public ref ITensor<TNumber> Gradient => ref _gradient;

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
        Value = _value.Add(amount);
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
            _value.Dispose();
            _gradient.Dispose();
        }
    }
}