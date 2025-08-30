// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Iterators;

internal static class BinaryOpPlanHelper
{
    public static DimRange[] GetDimRanges<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var outRank = result.Shape.Rank;

        var leftDimsPadded = PadShape(left.Shape.Dimensions, outRank);
        var leftStridesPadded = PadStrides(left.Shape.Strides, outRank);
        var rightDimsPadded = PadShape(right.Shape.Dimensions, outRank);
        var rightStridesPadded = PadStrides(right.Shape.Strides, outRank);
        var outDimsPadded = PadShape(result.Shape.Dimensions, outRank);
        var outStridesPadded = PadStrides(result.Shape.Strides, outRank);

        var merged = new List<DimRange>();

        var dim = outRank - 1;

        while (dim >= 0)
        {
            var size = outDimsPadded[dim];

            var sLeft = leftDimsPadded[dim] == 1 ? 0 : leftStridesPadded[dim];
            var sRight = rightDimsPadded[dim] == 1 ? 0 : rightStridesPadded[dim];
            var sOut = outStridesPadded[dim];

            var mergedSize = size;
            var moveDim = dim - 1;

            while (moveDim >= 0)
            {
                var nextSize = outDimsPadded[moveDim];
                var nextLeftDim = leftDimsPadded[moveDim];
                var nextRightDim = rightDimsPadded[moveDim];

                var nextLeftStride = nextLeftDim == 1 ? 0 : leftStridesPadded[moveDim];
                var nextRightStride = nextRightDim == 1 ? 0 : rightStridesPadded[moveDim];
                var nextOutStride = outStridesPadded[moveDim];

                var aOk = sLeft == nextLeftStride * nextSize || sLeft == 0 || nextLeftStride == 0;
                var bOk = sRight == nextRightStride * nextSize || sRight == 0 || nextRightStride == 0;
                var outOk = sOut == nextOutStride * nextSize;

                if (!aOk || !bOk || !outOk)
                {
                    break;
                }

                mergedSize *= nextSize;
                sLeft = sLeft != 0 ? sLeft : nextLeftStride;
                sRight = sRight != 0 ? sRight : nextRightStride;

                moveDim--;
            }

            merged.Add(
                new DimRange
                {
                    Extent = mergedSize,
                    StrideLeft = sLeft,
                    StrideRight = sRight,
                    StrideOut = sOut
                });

            dim = moveDim;
        }

        merged.Reverse();

        return merged.ToArray();
    }

    private static int[] PadShape(int[] dims, int rankWanted)
    {
        var diff = rankWanted - dims.Length;
        if (diff <= 0)
        {
            return dims;
        }

        var padded = new int[rankWanted];
        for (var i = 0; i < diff; i++)
        {
            padded[i] = 1;
        }

        for (var i = 0; i < dims.Length; i++)
        {
            padded[diff + i] = dims[i];
        }

        return padded;
    }

    private static long[] PadStrides(long[] strides, int rankWanted)
    {
        var diff = rankWanted - strides.Length;
        if (diff <= 0)
        {
            return strides;
        }

        var padded = new long[rankWanted];
        for (var i = 0; i < diff; i++)
        {
            padded[i] = 0;
        }

        for (var i = 0; i < strides.Length; i++)
        {
            padded[diff + i] = strides[i];
        }

        return padded;
    }
}