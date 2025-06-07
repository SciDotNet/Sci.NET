// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedPermutationKernels : IPermutationKernels
{
    public unsafe void Permute<TNumber>(ITensor<TNumber> source, ITensor<TNumber> result, int[] permutation)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var sourceBlock = (SystemMemoryBlock<TNumber>)source.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;
        var rank = permutation.Length;
        var dimsSource = source.Shape.Dimensions;
        var srcStrides = source.Shape.Strides;
        var dstStrides = result.Shape.Strides;
        var destDims = new int[rank];

        for (var j = 0; j < rank; j++)
        {
            destDims[j] = dimsSource[permutation[j]];
        }

        var permutedSrcStrides = new long[rank];
        for (var j = 0; j < rank; j++)
        {
            permutedSrcStrides[j] = srcStrides[permutation[j]];
        }

        var lastAxisIsFullyContiguous = permutation[rank - 1] == rank - 1;
        const int axisParallel = 0; // TODO - Figure out which axis should be parallelized
        var orderedAxes = new int[rank];
        orderedAxes[0] = axisParallel;

        var pos = 1;
        for (var a = 0; a < rank; a++)
        {
            if (a == axisParallel || a == rank - 1)
            {
                continue;
            }

            orderedAxes[pos++] = a;
        }

        orderedAxes[pos] = rank - 1;

        var srcPtrBase = sourceBlock.Pointer;
        var dstPtrBase = resultBlock.Pointer;

        _ = Parallel.For(
            0,
            destDims[axisParallel],
            baseIdx =>
            {
                var baseSrcOffset = baseIdx * permutedSrcStrides[axisParallel];
                var baseDstOffset = baseIdx * dstStrides[axisParallel];

                RecursiveCopy(
                    depth: 1,
                    currentSrcOffset: baseSrcOffset,
                    currentDstOffset: baseDstOffset,
                    orderedAxes: orderedAxes,
                    dstDims: destDims,
                    dstStrides: dstStrides,
                    permutedSrcStrides: permutedSrcStrides,
                    srcPtrBase: srcPtrBase,
                    dstPtrBase: dstPtrBase,
                    lastAxisCopyAsMemcpy: lastAxisIsFullyContiguous);
            });
    }

    private static unsafe void RecursiveCopy<TNumber>(
        int depth,
        long currentSrcOffset,
        long currentDstOffset,
        int[] orderedAxes,
        int[] dstDims,
        long[] dstStrides,
        long[] permutedSrcStrides,
        TNumber* srcPtrBase,
        TNumber* dstPtrBase,
        bool lastAxisCopyAsMemcpy)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var rank = orderedAxes.Length;
        var axis = orderedAxes[depth];

        if (depth < rank - 1)
        {
            var strideSrc = permutedSrcStrides[axis];
            var strideDst = dstStrides[axis];
            var axisDim = dstDims[axis];

            for (var i = 0; i < axisDim; i++)
            {
                var newSrcOff = currentSrcOffset + (i * strideSrc);
                var newDstOff = currentDstOffset + (i * strideDst);

                RecursiveCopy(
                    depth + 1,
                    newSrcOff,
                    newDstOff,
                    orderedAxes,
                    dstDims,
                    dstStrides,
                    permutedSrcStrides,
                    srcPtrBase,
                    dstPtrBase,
                    lastAxisCopyAsMemcpy);
            }
        }
        else
        {
            var copyCount = dstDims[axis];
            var srcStride = permutedSrcStrides[axis];
            var dstStride = dstStrides[axis];
            var sourcePtr = srcPtrBase + currentSrcOffset;
            var destinationPtr = dstPtrBase + currentDstOffset;

            if (lastAxisCopyAsMemcpy && srcStride == 1 && dstStride == 1)
            {
                var byteCount = (ulong)copyCount * (ulong)sizeof(TNumber);
                Buffer.MemoryCopy(sourcePtr, destinationPtr, byteCount, byteCount);
            }
            else
            {
                for (var k = 0; k < copyCount; k++)
                {
                    *destinationPtr = *sourcePtr;
                    destinationPtr++;
                    sourcePtr += srcStride;
                }
            }
        }
    }
}