// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using Sci.NET.Mathematics.Tensors.Backends;
using Sci.NET.Mathematics.Tensors.Manipulation;
using Sci.NET.Mathematics.Tensors.Pointwise;

namespace Sci.NET.Mathematics.Tensors.LinearAlgebra;

/// <summary>
/// Provides extension methods for performing a tensor contraction.
/// </summary>
[PublicAPI]
public static class ContractionExtensions
{
    /// <summary>
    /// Performs a tensor contraction on the specified tensors.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="leftIndices">The indices of the left operand to contract over.</param>
    /// <param name="rightIndices">The indices of the right operand to contract over.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the contraction operation.</returns>
    /// <exception cref="ArgumentException">Throws when an argument is invalid.</exception>
    public static ITensor<TNumber> Contract<TNumber>(
        this ITensor<TNumber> left,
        ITensor<TNumber> right,
        int[] leftIndices,
        int[] rightIndices)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var contractedSize = 1;
        if (leftIndices.Length != rightIndices.Length)
        {
            throw new ArgumentException("The number of left and right indices must be equal.");
        }

        for (var i = 0; i < leftIndices.Length; i++)
        {
            if (left.Dimensions[leftIndices[i]] != right.Dimensions[rightIndices[i]])
            {
                throw new ArgumentException("The dimensions of the left and right indices must be equal.");
            }

            contractedSize *= left.Dimensions[leftIndices[i]];
        }

        var leftContractedDims = DimListToBitset(leftIndices, left.Rank);
        var rightContractedDims = DimListToBitset(rightIndices, right.Rank);
        var leftPermutation = new List<int>();
        var rightPermutation = new List<int>();
        var resultShape = new List<int>();
        var leftSize = 1;
        var rightSize = 1;

        for (var i = 0; i < left.Rank; i++)
        {
            if (leftContractedDims[i])
            {
                continue;
            }

            leftPermutation.Add(i);
            leftSize *= left.Dimensions[i];
            resultShape.Add(left.Dimensions[i]);
        }

        leftPermutation.AddRange(leftIndices);
        rightPermutation.AddRange(rightIndices);

        for (var i = 0; i < right.Rank; i++)
        {
            if (rightContractedDims[i])
            {
                continue;
            }

            rightPermutation.Add(i);
            rightSize *= right.Dimensions[i];
            resultShape.Add(right.Dimensions[i]);
        }

        using var permutedLeft = left.Permute(leftPermutation.ToArray());
        using var permutedRight = right.Permute(rightPermutation.ToArray());
        using var reshapeLeft = permutedLeft.Reshape(leftSize, contractedSize);
        using var reshapeRight = permutedRight.Reshape(contractedSize, rightSize);

        var leftCopy = reshapeLeft.Handle.CopyToArray(reshapeLeft.ElementCount);
        var rightCopy = reshapeRight.Handle.CopyToArray(reshapeRight.ElementCount);

        _ = leftCopy;
        _ = rightCopy;

        using var mm = reshapeLeft.MatrixMultiply(reshapeRight);
        return mm.Reshape(resultShape.ToArray());
    }

    /// <inheritdoc cref="Contract{TNumber}"/>
    /// <remarks>An alias of <see cref="Contract{TNumber}"/>.</remarks>
    public static ITensor<TNumber> TensorDot<TNumber>(
        this ITensor<TNumber> left,
        ITensor<TNumber> right,
        int[] leftIndices,
        int[] rightIndices)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return Contract(left, right, leftIndices, rightIndices);
    }

    /// <summary>
    /// For vectors, the inner product of two <see cref="ITensor{TNumber}"/>s is calculated,
    /// for higher dimensions then the sum product over the last axes are calculated.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the inner product operation.</returns>
    /// <exception cref="ArgumentException">Throws when the operand shapes are incompatible with the
    /// inner product operation.</exception>
    [SuppressMessage("Style", "IDE0046:Convert to conditional expression", Justification = "Readability")]
    public static ITensor<TNumber> Inner<TNumber>(this ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Dimensions[^1] != right.Dimensions[^1])
        {
            throw new ArgumentException(
                "The last dimensions of the left and right operands must be equal.",
                nameof(right));
        }

        if (left.Rank == 0 || right.Rank == 1)
        {
            return left.ScalarProduct(right);
        }

        if (left.Rank == 1 && right.Rank == 1)
        {
            return TensorBackend.Instance.InnerProduct(left, right);
        }

        return left.TensorDot(right, new int[] { left.Rank - 1 }, new int[] { right.Rank - 1 });
    }

    private static bool[] DimListToBitset(int[] leftIndices, int leftRank)
    {
        var bits = new bool[leftRank];
        foreach (var i in leftIndices)
        {
            bits[i] = true;
        }

        return bits;
    }
}