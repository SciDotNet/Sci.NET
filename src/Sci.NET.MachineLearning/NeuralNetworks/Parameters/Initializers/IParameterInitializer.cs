// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.NeuralNetworks.Parameters.Initializers;

/// <summary>
/// An interface for parameter initializers.
/// </summary>
[PublicAPI]
public interface IParameterInitializer
{
    /// <summary>
    /// Initializes a parameter.
    /// </summary>
    /// <param name="shape">The shape of the parameter.</param>
    /// <param name="device">The device to store the <see cref="ITensor{TNumber}"/> data on.</param>
    /// <typeparam name="TNumber">The number type of the parameter.</typeparam>
    /// <returns>The initialized parameter.</returns>
    public ITensor<TNumber> Initialize<TNumber>(Shape shape, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>;
}