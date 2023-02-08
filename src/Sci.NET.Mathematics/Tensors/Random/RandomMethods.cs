// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using Sci.NET.Mathematics.Tensors.Backends;

namespace Sci.NET.Mathematics.Tensors.Random;

/// <summary>
/// Interface for random operations for an <see cref="ITensor{TNumber}"/>.
/// </summary>
[SuppressMessage(
    "Performance",
    "CA1822:Mark members as static",
    Justification = "This is required to be accessed through the TensorBackend.Instance.Random property.")]
public class RandomMethods
{
    /// <inheritdoc cref="IRandomBackendOperations.Uniform{TNumber}"/>
    [PublicAPI]
    public ITensor<TNumber> Uniform<TNumber>(Shape shape, TNumber min, TNumber max, long seed)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorBackend.Instance.Random.Uniform(shape, min, max, seed);
    }
}