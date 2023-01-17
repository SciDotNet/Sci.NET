// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Manipulation;

/// <summary>
/// Extension methods to permute a <see cref="ITensor{TNumber}"/>.
/// </summary>
[PublicAPI]
public static class PermutationExtensions
{
    /// <summary>
    /// Permutes the <see cref="ITensor{TNumber}"/>, rearranging the indices
    /// in the order passed as a parameter.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to permute.</param>
    /// <param name="permutation">The new order of the indices.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The permuted <see cref="ITensor{TNumber}"/>.</returns>
    /// <exception cref="ArgumentException">Throws when the <paramref name="permutation"/>
    /// indices are invalid.</exception>
    public static ITensor<TNumber> Permute<TNumber>(this ITensor<TNumber> tensor, int[] permutation)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (permutation.Distinct().Count() != tensor.Rank)
        {
            throw new ArgumentException("Permutation length must be equal to tensor rank.");
        }

        var permutedShape = new int[tensor.Rank];
        for (var i = 0; i < permutation.Length; i++)
        {
            if (permutation[i] < 0 || permutation[i] >= tensor.Rank)
            {
                throw new ArgumentException(
                    $"Permutation must contain all integers from 0 to Rank-1 (in this case {tensor.Rank - 1}).",
                    nameof(permutation));
            }

            permutedShape[i] = tensor.Dimensions[permutation[i]];
        }

        return new VirtualTensor<TNumber>(tensor, new Shape(permutedShape));
    }
}