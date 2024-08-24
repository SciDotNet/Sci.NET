// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Manipulation.Implementations;

internal class PermutationService : IPermutationService
{
    public ITensor<TNumber> Permute<TNumber>(ITensor<TNumber> tensor, int[] permutation, bool? overrideRequiresGradient = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (permutation.Distinct().Count() != tensor.Shape.Rank)
        {
            throw new ArgumentException("Permutation length must be equal to tensor rank.");
        }

        var permutedShape = new int[tensor.Shape.Rank];

        for (var i = 0; i < permutation.Length; i++)
        {
            if (permutation[i] < 0 || permutation[i] >= tensor.Shape.Rank)
            {
                throw new ArgumentException(
                    $"Permutation must contain all integers from 0 to Rank-1 (in this case {tensor.Shape.Rank - 1}).",
                    nameof(permutation));
            }

            permutedShape[i] = tensor.Shape[permutation[i]];
        }

        var result = new Tensor<TNumber>(new Shape(permutedShape), tensor.Backend, overrideRequiresGradient ?? tensor.RequiresGradient);

        result.Backend.Permutation.Permute(tensor, result, permutation);

        if (tensor.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                tensor,
                grad =>
                {
                    var inversePermutation = InversePermutation(permutation);
                    return grad.Permute(inversePermutation);
                });
        }

        return result;
    }

    private static int[] InversePermutation(int[] permutation)
    {
        var inverse = new int[permutation.Length];

        for (var i = 0; i < permutation.Length; i++)
        {
            inverse[permutation[i]] = i;
        }

        return inverse;
    }
}