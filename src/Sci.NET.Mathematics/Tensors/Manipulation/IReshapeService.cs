// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Manipulation;

/// <summary>
/// An interface providing methods to reshape <see cref="ITensor{TNumber}"/> instances.
/// </summary>
[PublicAPI]
public interface IReshapeService
{
    /// <summary>
    /// Reshapes a <see cref="ITensor{TNumber}"/> to a new shape.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to reshape.</param>
    /// <param name="shape">The new shape of the <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="overrideRequiresGradient">Whether the new <see cref="ITensor{TNumber}"/> requires a gradient.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The reshaped <see cref="ITensor{TNumber}"/>.</returns>
    /// <exception cref="ArgumentException">Throws if the new <paramref name="shape"/> is incompatible with
    /// the <see cref="ITensor{TNumber}"/>.</exception>
    public ITensor<TNumber> Reshape<TNumber>(ITensor<TNumber> tensor, Shape shape, bool? overrideRequiresGradient = null)
        where TNumber : unmanaged, INumber<TNumber>;
}