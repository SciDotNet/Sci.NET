// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.Manipulation.Implementations;

internal class PermutationService : IPermutationService
{
    private readonly IGradientAppenderService _gradientAppenderService;

    public PermutationService()
    {
        _gradientAppenderService = TensorServiceProvider.GetTensorOperationServiceProvider().GetGradientAppenderService();
    }

    public ITensor<TNumber> Permute<TNumber>(ITensor<TNumber> tensor, int[] permutation, bool? overrideRequiresGradient = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (tensor.Shape.IsScalar || tensor.Shape.IsVector)
        {
            throw new InvalidShapeException($"{tensor} must not be rank {tensor.Shape.Rank}. Permutation is not supported for scalar or vector tensors.");
        }

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

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            overrideRequiresGradient,
            grad => grad.Permute(permutation));

        return result;
    }
}