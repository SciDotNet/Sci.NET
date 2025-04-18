// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Exceptions;
using Sci.NET.Mathematics.Tensors.Manipulation;
using Sci.NET.Mathematics.Tensors.Pointwise;
using Sci.NET.Mathematics.Tensors.Reduction;

namespace Sci.NET.Mathematics.Tensors.LinearAlgebra.Implementations;

internal class ContractionService : IContractionService
{
    private readonly IDeviceGuardService _guardService;
    private readonly IMatrixMultiplicationService _matrixMultiplicationService;
    private readonly IArithmeticService _arithmeticService;
    private readonly IReshapeService _reshapeService;
    private readonly IPermutationService _permutationService;
    private readonly IReductionService _reductionService;
    private readonly IGradientAppenderService _gradientAppenderService;

    public ContractionService()
    {
        _guardService = TensorServiceProvider.GetTensorOperationServiceProvider().GetDeviceGuardService();
        _matrixMultiplicationService = TensorServiceProvider.GetTensorOperationServiceProvider().GetMatrixMultiplicationService();
        _permutationService = TensorServiceProvider.GetTensorOperationServiceProvider().GetPermutationService();
        _reshapeService = TensorServiceProvider.GetTensorOperationServiceProvider().GetReshapeService();
        _arithmeticService = TensorServiceProvider.GetTensorOperationServiceProvider().GetArithmeticService();
        _reductionService = TensorServiceProvider.GetTensorOperationServiceProvider().GetReductionService();
        _gradientAppenderService = TensorServiceProvider.GetTensorOperationServiceProvider().GetGradientAppenderService();
    }

    [SuppressMessage("Style", "IDE0045:Convert to conditional expression", Justification = "Readability")]
    public ITensor<TNumber> Contract<TNumber>(
        ITensor<TNumber> left,
        ITensor<TNumber> right,
        int[] leftIndices,
        int[] rightIndices,
        bool? overrideRequiresGradient = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _ = _guardService.GuardBinaryOperation(left.Device, right.Device);

        var contractedSize = 1;

        ArgumentOutOfRangeException.ThrowIfNotEqual(leftIndices.Length, rightIndices.Length);

        for (var i = 0; i < leftIndices.Length; i++)
        {
            var leftDimSize = left.Shape[leftIndices[i]];
            var rightDimSize = right.Shape[rightIndices[i]];

            if (leftDimSize != rightDimSize)
            {
                if (leftDimSize == 1)
                {
                    left = _reductionService.Sum(left, new[] { leftIndices[i] }, true);
                }
                else if (rightDimSize == 1)
                {
                    right = _reductionService.Sum(right, new[] { rightIndices[i] }, true);
                }
                else
                {
                    throw new InvalidIndicesException($"The contracted dimensions need to match, but the the first has size {leftDimSize} at dimension {leftIndices[i]} and the second has {rightDimSize} at dimension {rightIndices[i]}.");
                }
            }
            else
            {
                contractedSize *= leftDimSize;
            }
        }

        var leftIsContracted = DimListToBitset(leftIndices, left.Shape.Rank);
        var rightIsContracted = DimListToBitset(rightIndices, right.Shape.Rank);
        var leftPermutation = new List<int>(left.Shape.Rank);
        var rightPermutation = new List<int>(right.Shape.Rank);
        var resultShape = new List<int>(left.Shape.Rank + right.Shape.Rank - leftIndices.Length);
        var nonContractedLeft = 1;
        var nonContractedRight = 1;

        for (var i = 0; i < left.Shape.Rank; i++)
        {
            if (!leftIsContracted[i])
            {
                leftPermutation.Add(i);
                nonContractedLeft *= left.Shape[i];
                resultShape.Add(left.Shape[i]);
            }
        }

        leftPermutation.AddRange(leftIndices);
        rightPermutation.AddRange(rightIndices);

        for (var i = 0; i < right.Shape.Rank; i++)
        {
            if (!rightIsContracted[i])
            {
                rightPermutation.Add(i);
                nonContractedRight *= right.Shape[i];
                resultShape.Add(right.Shape[i]);
            }
        }

        using var leftPermuted = _permutationService.Permute(left, leftPermutation.ToArray(), overrideRequiresGradient: false);
        using var rightPermuted = _permutationService.Permute(right, rightPermutation.ToArray(), overrideRequiresGradient: false);
        using var leftReshaped = _reshapeService.Reshape(leftPermuted, new Shape(nonContractedLeft, contractedSize)).ToMatrix(false);
        using var rightReshaped = _reshapeService.Reshape(rightPermuted, new Shape(contractedSize, nonContractedRight)).ToMatrix(false);
        using var mmResult = _matrixMultiplicationService.MatrixMultiply(leftReshaped, rightReshaped, overrideRequiresGradient: false);
        var result = _reshapeService.Reshape(mmResult, new Shape(resultShape.ToArray()), overrideRequiresGradient: false);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            left,
            right,
            overrideRequiresGradient,
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

                var contractionService = TensorServiceProvider.GetTensorOperationServiceProvider().GetContractionService();
                var permutationService = TensorServiceProvider.GetTensorOperationServiceProvider().GetPermutationService();
                var leftInd = Enumerable.Range(nonContractedBAxes.Length, grad.Shape.Rank - nonContractedBAxes.Length).ToList();
                var permuteAxes = nonContractedAAxes.Concat(leftIndices).ToArray();
                var aGrad = contractionService.Contract(grad, right, leftInd.ToArray(), nonContractedBAxes.ToArray(), overrideRequiresGradient: false);
                return permutationService.Permute(aGrad, permuteAxes, overrideRequiresGradient: false);
            },
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

                var contractionService = TensorServiceProvider.GetTensorOperationServiceProvider().GetContractionService();
                var permutationService = TensorServiceProvider.GetTensorOperationServiceProvider().GetPermutationService();
                var permuteAxes = new int[right.Shape.Rank];
                var currentIndex = 0;
                foreach (var axis in nonContractedBAxes)
                {
                    permuteAxes[axis] = currentIndex++;
                }

                foreach (var axis in rightIndices)
                {
                    permuteAxes[axis] = currentIndex++;
                }

                var rightInd = Enumerable.Range(0, grad.Shape.Rank - nonContractedAAxes.Length).ToList();
                var bGrad = contractionService.Contract(grad, left, rightInd.ToArray(), nonContractedAAxes.ToArray(), overrideRequiresGradient: false);

                return permutationService.Permute(bGrad, permuteAxes, overrideRequiresGradient: false);
            });

        return result;
    }

    public Scalar<TNumber> Inner<TNumber>(Vector<TNumber> left, Vector<TNumber> right, bool? overrideRequiresGradient = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = _guardService.GuardBinaryOperation(left.Device, right.Device);

        ArgumentOutOfRangeException.ThrowIfNotEqual(left.Shape[^1], right.Shape[^1]);
        InvalidShapeException.ThrowIfNotOfRank(left, 1);
        InvalidShapeException.ThrowIfNotOfRank(right, 1);

        var result = new Scalar<TNumber>(backend);
        backend.LinearAlgebra.InnerProduct(left, right, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            left,
            right,
            overrideRequiresGradient,
            grad =>
            {
                var gradVector = right.Multiply(grad);
                return gradVector.AsGradient();
            },
            grad =>
            {
                var gradVector = left.Multiply(grad);
                return gradVector.AsGradient();
            });

        return result;
    }

    public ITensor<TNumber> Dot<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
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
            return left.Contract(right, [left.Shape.Rank - 1], [right.Shape.Rank - 1]);
        }

        return left.Contract(right, [left.Shape.Rank - 1], [right.Shape.Rank - 2]);
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
}