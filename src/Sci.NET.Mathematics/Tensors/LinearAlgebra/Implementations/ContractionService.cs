// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Exceptions;
using Sci.NET.Mathematics.Tensors.Manipulation;
using Sci.NET.Mathematics.Tensors.Pointwise;

namespace Sci.NET.Mathematics.Tensors.LinearAlgebra.Implementations;

internal class ContractionService : IContractionService
{
    private readonly IDeviceGuardService _guardService;
    private readonly IMatrixMultiplicationService _matrixMultiplicationService;
    private readonly IArithmeticService _arithmeticService;
    private readonly IReshapeService _reshapeService;
    private readonly IPermutationService _permutationService;

    public ContractionService(ITensorOperationServiceProvider provider)
    {
        _guardService = provider.GetDeviceGuardService();
        _matrixMultiplicationService = provider.GetMatrixMultiplicationService();
        _permutationService = provider.GetPermutationService();
        _reshapeService = provider.GetReshapeService();
        _arithmeticService = provider.GetArithmeticService();
    }

    public ITensor<TNumber> Contract<TNumber>(
        ITensor<TNumber> left,
        ITensor<TNumber> right,
        int[] leftIndices,
        int[] rightIndices)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);

        var contractedSize = 1;

        ArgumentOutOfRangeException.ThrowIfNotEqual(leftIndices.Length, rightIndices.Length);

        for (var i = 0; i < leftIndices.Length; i++)
        {
            ArgumentOutOfRangeException.ThrowIfNotEqual(left.Shape[leftIndices[i]], right.Shape[rightIndices[i]]);

            contractedSize *= left.Shape[leftIndices[i]];
        }

        var leftContractedDims = DimListToBitset(leftIndices, left.Shape.Rank);
        var rightContractedDims = DimListToBitset(rightIndices, right.Shape.Rank);
        var leftPermutation = new List<int>();
        var rightPermutation = new List<int>();
        var resultShape = new List<int>();
        var leftSize = 1;
        var rightSize = 1;

        for (var i = 0; i < left.Shape.Rank; i++)
        {
            if (leftContractedDims[i])
            {
                continue;
            }

            leftPermutation.Add(i);
            leftSize *= left.Shape[i];
            resultShape.Add(left.Shape[i]);
        }

        leftPermutation.AddRange(leftIndices);
        rightPermutation.AddRange(rightIndices);

        for (var i = 0; i < right.Shape.Rank; i++)
        {
            if (rightContractedDims[i])
            {
                continue;
            }

            rightPermutation.Add(i);
            rightSize *= right.Shape[i];
            resultShape.Add(right.Shape[i]);
        }

        using var permutedLeft = _permutationService.Permute(left, leftPermutation.ToArray());
        using var permutedRight = _permutationService.Permute(right, rightPermutation.ToArray());
        using var reshapeLeft = _reshapeService.Reshape(permutedLeft, new Shape(leftSize, contractedSize));
        using var reshapeLeftMatrix = reshapeLeft.ToMatrix(requiresGradient: false);
        using var reshapeRight = _reshapeService.Reshape(permutedRight, new Shape(contractedSize, rightSize));
        using var reshapeRightMatrix = reshapeRight.ToMatrix(requiresGradient: false);
        using var mm = _matrixMultiplicationService.MatrixMultiply(reshapeLeftMatrix, reshapeRightMatrix, requiresGradient: false);
        var result = _reshapeService.Reshape(mm, new Shape(resultShape.ToArray()), requiresGradient: false).RecreateWithGradient();

        if (left.RequiresGradient)
        {
            result.AddParent(
                left,
                grad =>
                {
                    var nonContractedAAxes = Enumerable
                        .Range(0, left.Shape.Rank)
                        .Where(i => !leftIndices.Contains(i))
                        .ToArray();
                    var nonContractedBAxes = Enumerable
                        .Range(0, right.Shape.Rank)
                        .Where(i => !rightIndices.Contains(i))
                        .ToArray();

                    var leftInd = Enumerable.Range(nonContractedBAxes.Length, grad.Shape.Rank - nonContractedBAxes.Length).ToList();
                    var aGrad = grad.Contract(right, leftInd.ToArray(), nonContractedBAxes.ToArray());
                    var permuteAxes = nonContractedAAxes.Concat(leftIndices).ToArray();

                    return aGrad.Permute(permuteAxes);
                });
        }

        if (right.RequiresGradient)
        {
            result.AddParent(
                right,
                _ =>
                {
                    throw new NotImplementedException();
                });
        }

        return result;
    }

    public Scalar<TNumber> Inner<TNumber>(Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);

        ArgumentOutOfRangeException.ThrowIfNotEqual(left.Shape[^1], right.Shape[^1]);
        InvalidShapeException.ThrowIfNotOfRank(left, 1);
        InvalidShapeException.ThrowIfNotOfRank(right, 1);

        var result = new Scalar<TNumber>(left.Backend);
        left.Backend.LinearAlgebra.InnerProduct(left, right, result);

        return result;
    }

    public ITensor<TNumber> Dot<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);

        if (left.IsVector() && right.IsVector())
        {
            return Inner(left.ToVector(), right.ToVector());
        }

        if (left.IsScalar() || right.IsScalar())
        {
            return _arithmeticService.Multiply(left.ToScalar(), right.ToScalar());
        }

        if (left.IsMatrix() && right.IsMatrix())
        {
            return _matrixMultiplicationService.MatrixMultiply(left.ToMatrix(), right.ToMatrix());
        }

        if (right.IsVector())
        {
            return left.Contract(
                right,
                new int[] { left.Shape.Rank - 1 },
                new int[] { right.Shape.Rank - 1 });
        }

        return left.Contract(
            right,
            new int[] { left.Shape.Rank - 1 },
            new int[] { right.Shape.Rank - 2 });
    }

    private static bool[] DimListToBitset(IEnumerable<int> leftIndices, int leftRank)
    {
        var bits = new bool[leftRank];

        foreach (var i in leftIndices)
        {
            bits[i] = true;
        }

        return bits;
    }

    private static int[] InversePermutation(int[] perm)
    {
        var inversePerm = new int[perm.Length];

        for (int i = 0; i < perm.Length; i++)
        {
            inversePerm[perm[i]] = i;
        }

        return inversePerm;
    }
}