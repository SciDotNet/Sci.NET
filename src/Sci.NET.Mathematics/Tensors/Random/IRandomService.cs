// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Mathematics.Tensors.Random;

/// <summary>
/// A service for generating random <see cref="ITensor{TNumber}"/>s.
/// </summary>
[PublicAPI]
public interface IRandomService
{
    /// <summary>
    /// Generates a random <see cref="ITensor{TNumber}"/> with values drawn from a normal distribution.
    /// </summary>
    /// <param name="shape">The shape of the <see cref="ITensor{TNumber}"/> to generate.</param>
    /// <param name="min">The minimum value of the distribution.</param>
    /// <param name="max">The maximum value of the distribution.</param>
    /// <param name="seed">The seed for the random number generator.</param>
    /// <param name="device">The device to generate the <see cref="ITensor{TNumber}"/> on.</param>
    /// <typeparam name="TNumber">The type of the <see cref="ITensor{TNumber}"/> to generate.</typeparam>
    /// <returns>A random <see cref="ITensor{TNumber}"/> with values drawn from a normal distribution.</returns>
    public ITensor<TNumber> Uniform<TNumber>(Shape shape, TNumber min, TNumber max, int? seed = null, IDevice? device = null)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Generates a random <see cref="ITensor{TNumber}"/> with values drawn from a normal distribution.
    /// </summary>
    /// <param name="shape">The shape of the <see cref="ITensor{TNumber}"/> to generate.</param>
    /// <param name="min">The minimum value of the distribution.</param>
    /// <param name="max">The maximum value of the distribution.</param>
    /// <param name="seed">The seed for the random number generator.</param>
    /// <typeparam name="TNumber">The type of the <see cref="ITensor{TNumber}"/> to generate.</typeparam>
    /// <typeparam name="TDevice">The type of device to generate the <see cref="ITensor{TNumber}"/> on.</typeparam>
    /// <returns>A random <see cref="ITensor{TNumber}"/> with values drawn from a normal distribution.</returns>
    public ITensor<TNumber> Uniform<TNumber, TDevice>(Shape shape, TNumber min, TNumber max, int? seed = null)
        where TNumber : unmanaged, INumber<TNumber>
        where TDevice : IDevice, new();
}