// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Manipulation;

/// <summary>
/// An interface providing methods to permute <see cref="ITensor{TNumber}"/> instances.
/// </summary>
public interface IPermutationService
{
    /// <summary>
    /// Permutes the <see cref="ITensor{TNumber}"/>, rearranging the indices
    /// in the order passed as a parameter.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to permute.</param>
    /// <param name="permutation">The new order of the indices.</param>
    /// <param name="overrideRequiresGradient">When not <see langword="null"/>, overrides whether the resulting tensor requires gradient.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The permuted <see cref="ITensor{TNumber}"/>.</returns>
    /// <exception cref="ArgumentException">Throws when the <paramref name="permutation"/>
    /// indices are invalid.</exception>
    public ITensor<TNumber> Permute<TNumber>(ITensor<TNumber> tensor, int[] permutation, bool? overrideRequiresGradient = null)
        where TNumber : unmanaged, INumber<TNumber>;
}