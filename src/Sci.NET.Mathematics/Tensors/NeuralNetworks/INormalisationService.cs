// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.NeuralNetworks;

/// <summary>
/// Service for tensor normalisation.
/// </summary>
[PublicAPI]
public interface INormalisationService
{
    /// <summary>
    /// Normalises the input tensor using batch normalisation.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="scale">The weight tensor.</param>
    /// <param name="bias">The bias tensor.</param>
    /// <typeparam name="TNumber">The type of the tensor.</typeparam>
    /// <returns>The normalised tensor.</returns>
    public Matrix<TNumber> BatchNorm1dForward<TNumber>(Matrix<TNumber> input, Vector<TNumber> scale, Vector<TNumber> bias)
        where TNumber : unmanaged, IRootFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Clips the values of the <see cref="ITensor{TNumber}"/> to the specified range.
    /// </summary>
    /// <param name="tensor">The the <see cref="ITensor{TNumber}"/> to operate on.</param>
    /// <param name="min">The minimum value to clip to.</param>
    /// <param name="max">The maximum value to clip to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The clipped <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Clip<TNumber>(ITensor<TNumber> tensor, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>;
}