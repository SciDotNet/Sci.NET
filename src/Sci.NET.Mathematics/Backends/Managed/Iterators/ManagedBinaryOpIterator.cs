// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Performance;
using Sci.NET.Mathematics.Backends.Iterators;
using Sci.NET.Mathematics.Backends.Managed.BinaryOps;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed.Iterators;

internal class ManagedBinaryOpIterator<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly DimRange[] _ranges;
    private readonly bool _allUnitStrideNoBroadcast;
    private readonly long _total;

    public ManagedBinaryOpIterator(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
    {
        _ranges = BuildDimRanges(left, right, result);
        _allUnitStrideNoBroadcast = CheckAllUnitStrideNoBroadcast(_ranges);
        _total = 1;

        foreach (var range in _ranges)
        {
            _total *= range.Extent;
        }
    }

    public unsafe void Apply<TOp>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TOp : IManagedBinaryTensorOp
    {
        var leftPointer = left.Memory.ToPointer();
        var rightPointer = right.Memory.ToPointer();
        var resultPointer = result.Memory.ToPointer();

        if (_allUnitStrideNoBroadcast)
        {
            ManagedBinaryOpIterator.ApplyBlock1D<TNumber, TOp>(
                leftPointer,
                rightPointer,
                resultPointer,
                _total,
                1,
                1,
                1);
            return;
        }

        if (_ranges.Length == 0)
        {
            var leftItem = Unsafe.Read<TNumber>(leftPointer);
            var rightItem = Unsafe.Read<TNumber>(rightPointer);
            Unsafe.Write(resultPointer, TOp.Invoke(leftItem, rightItem));
            return;
        }

        if (_ranges.Length == 1)
        {
            var range = _ranges[0];
            ManagedBinaryOpIterator.ApplyBlock1D<TNumber, TOp>(
                leftPointer,
                rightPointer,
                resultPointer,
                range.Extent,
                range.StrideLeft,
                range.StrideRight,
                range.StrideResult);
            return;
        }

        if (_ranges.Length == 2)
        {
            var outer = _ranges[0];
            var inner = _ranges[1];
            var nOuter = outer.Extent;
            var strideLeftOuter = outer.StrideLeft;
            var strideRightOuter = outer.StrideRight;
            var strideResultOuter = outer.StrideResult;

            _ = LazyParallelExecutor.For(
                0,
                nOuter,
                ManagedTensorBackend.ParallelizationThreshold,
                i =>
                {
                    var baseLeft = leftPointer + (i * strideLeftOuter);
                    var baseRight = rightPointer + (i * strideRightOuter);
                    var baseResult = resultPointer + (i * strideResultOuter);

                    ManagedBinaryOpIterator.ApplyBlock1D<TNumber, TOp>(
                        baseLeft,
                        baseRight,
                        baseResult,
                        inner.Extent,
                        inner.StrideLeft,
                        inner.StrideRight,
                        inner.StrideResult);
                });
            return;
        }

        _ = LazyParallelExecutor.For(
            0,
            _total,
            ManagedTensorBackend.ParallelizationThreshold,
            idx =>
            {
                long leftOffset = 0, rightOffset = 0, resultOffset = 0;
                for (var rangeIdx = _ranges.Length - 1; rangeIdx >= 0; rangeIdx--)
                {
                    var range = _ranges[rangeIdx];
                    var offset = idx % range.Extent;
                    idx /= range.Extent;
                    leftOffset += offset * range.StrideLeft;
                    rightOffset += offset * range.StrideRight;
                    resultOffset += offset * range.StrideResult;
                }

                var leftItem = Unsafe.Read<TNumber>(leftPointer + leftOffset);
                var rightItem = Unsafe.Read<TNumber>(rightPointer + rightOffset);
                Unsafe.Write(resultPointer + resultOffset, TOp.Invoke(leftItem, rightItem));
            });
    }

    [MethodImpl(ImplementationOptions.FastPath)]
    private static bool CheckAllUnitStrideNoBroadcast(DimRange[] ranges)
    {
        foreach (var range in ranges)
        {
            if (range.StrideResult != 1)
            {
                return false;
            }

            if (range.StrideLeft == 0 || range.StrideRight == 0)
            {
                return false;
            }

            if (range.StrideLeft != 1 || range.StrideRight != 1)
            {
                return false;
            }
        }

        return true;
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
            var leftStride = leftDimsPadded[dim] == 1 ? 0 : leftStridesPadded[dim];
            var rightStride = rightDimsPadded[dim] == 1 ? 0 : rightStridesPadded[dim];
            var resultStride = outStridesPadded[dim];
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
                var leftOk = leftStride == nextLeftStride * nextSize || leftStride == 0 || nextLeftStride == 0;
                var rightOk = rightStride == nextRightStride * nextSize || rightStride == 0 || nextRightStride == 0;
                var resultOk = resultStride == nextOutStride * nextSize;

                if (!leftOk || !rightOk || !resultOk)
                {
                    break;
                }

                mergedSize *= nextSize;
                leftStride = leftStride != 0 ? leftStride : nextLeftStride;
                rightStride = rightStride != 0 ? rightStride : nextRightStride;

                moveDim--;
            }

            merged.Add(
                new DimRange
                {
                    Extent = mergedSize,
                    StrideLeft = leftStride,
                    StrideRight = rightStride,
                    StrideResult = resultStride
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