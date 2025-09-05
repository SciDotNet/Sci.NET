// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Mathematics.Backends.Iterators;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed.Iterators;

internal class ManagedBinaryOpIterator<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly DimRange[] _dimRanges;

    public ManagedBinaryOpIterator(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
    {
        _dimRanges = BuildDimRanges(left, right, result);
    }

    /// <summary>
    /// Iterates over the tensors and applies the given action to each element.
    /// </summary>
    /// <param name="action">The action to apply to each element.</param>
    public void Apply(Action<long, long, long> action)
    {
        var rank = _dimRanges.Length;

        if (rank == 0)
        {
            action(0, 0, 0);
        }
        else if (rank == 1)
        {
            Apply1D(action);
        }
        else if (rank == 2)
        {
            Apply2D(action);
        }
        else
        {
            ApplyND(action);
        }
    }

    private static DimRange[] BuildDimRanges(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
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
                    StrideResult = sOut
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

    private void Apply1D(Action<long, long, long> action)
    {
        var d0 = _dimRanges[0];
        var extent = d0.Extent;
        var sL = d0.StrideLeft;
        var sR = d0.StrideRight;
        var sO = d0.StrideResult;

        _ = LazyParallelExecutor.For(
            0,
            extent,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var offLeft = i * sL;
                var offRight = i * sR;
                var offOut = i * sO;

                action(offLeft, offRight, offOut);
            });
    }

    private void Apply2D(Action<long, long, long> action)
    {
        var dim0 = _dimRanges[0];
        var dim1 = _dimRanges[1];
        var extent0 = dim0.Extent;
        var extent1 = dim1.Extent;
        var leftStrideDim0 = dim0.StrideLeft;
        var rightStrideDim0 = dim0.StrideRight;
        var resultStrideDim0 = dim0.StrideResult;
        var leftStrideDim1 = dim1.StrideLeft;
        var rightStrideDim1 = dim1.StrideRight;
        var resultStrideDim1 = dim1.StrideResult;
        var total = extent0 * extent1;

        _ = LazyParallelExecutor.For(
            0,
            total,
            ManagedTensorBackend.ParallelizationThreshold,
            idx =>
            {
                var i = idx / extent1;
                var j = idx % extent1;

                var baseLeft = i * leftStrideDim0;
                var baseRight = i * rightStrideDim0;
                var baseOut = i * resultStrideDim0;

                var offLeft = baseLeft + (j * leftStrideDim1);
                var offRight = baseRight + (j * rightStrideDim1);
                var offOut = baseOut + (j * resultStrideDim1);

                action(offLeft, offRight, offOut);
            });
    }

    private void ApplyND(Action<long, long, long> action)
    {
        long totalElements = 1;
        for (var d = 0; d < _dimRanges.Length; d++)
        {
            totalElements *= _dimRanges[d].Extent;
        }

        _ = LazyParallelExecutor.For(
            0,
            totalElements,
            ManagedTensorBackend.ParallelizationThreshold,
            idx =>
            {
                var offsetLeft = 0L;
                var offsetRight = 0L;
                var offsetOut = 0L;

                var tmp = idx;

                for (var dim = _dimRanges.Length - 1; dim >= 0; dim--)
                {
                    var ext = _dimRanges[dim].Extent;
                    var coordinate = tmp % ext;
                    tmp /= ext;

                    offsetLeft += coordinate * _dimRanges[dim].StrideLeft;
                    offsetRight += coordinate * _dimRanges[dim].StrideRight;
                    offsetOut += coordinate * _dimRanges[dim].StrideResult;
                }

                action(offsetLeft, offsetRight, offsetOut);
            });
    }
}