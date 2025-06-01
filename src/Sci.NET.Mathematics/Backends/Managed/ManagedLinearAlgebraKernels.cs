// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Numerics.Intrinsics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedLinearAlgebraKernels : ILinearAlgebraKernels
{
    public unsafe void MatrixMultiply<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Memory;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Memory;
        var leftMemoryBlockPtr = leftMemoryBlock.ToPointer();
        var rightMemoryBlockPtr = rightMemoryBlock.ToPointer();
        var resultMemoryBlockPtr = resultMemoryBlock.ToPointer();
        var leftRows = left.Rows;
        var rightColumns = right.Columns;
        var leftColumns = left.Columns;
        const int iBlock = 128;
        const int jBlock = 16;
        var avxSupported = Vector.IsHardwareAccelerated && Avx.IsSupported;

        if (typeof(TNumber) == typeof(float) && avxSupported)
        {
            MatrixMultiply8x8AvxFloatParallel(
                (float*)leftMemoryBlockPtr,
                (float*)rightMemoryBlockPtr,
                (float*)resultMemoryBlockPtr,
                leftRows,
                rightColumns,
                leftColumns);
            return;
        }

        if (typeof(TNumber) == typeof(double) && avxSupported)
        {
            MatrixMultiply8X8AvxDoubleParallel(
                (double*)leftMemoryBlockPtr,
                (double*)rightMemoryBlockPtr,
                (double*)resultMemoryBlockPtr,
                leftRows,
                rightColumns,
                leftColumns);
            return;
        }

        LazyParallelExecutor.ForBlocked(
            0,
            leftRows,
            0,
            rightColumns,
            iBlock,
            jBlock,
            (i0, j0) =>
            {
                var iMax = Math.Min(i0 + iBlock, leftRows);
                var jMax = Math.Min(j0 + jBlock, rightColumns);

                for (var i = i0; i < iMax; ++i)
                {
                    for (var j = j0; j < jMax; ++j)
                    {
                        var sum = TNumber.Zero;
                        for (int k = 0; k < leftColumns; ++k)
                        {
                            sum += leftMemoryBlockPtr[(i * leftColumns) + k] * rightMemoryBlockPtr[(k * rightColumns) + j];
                        }

                        resultMemoryBlockPtr[(i * rightColumns) + j] = sum;
                    }
                }
            });
    }

    public unsafe void InnerProduct<TNumber>(Tensors.Vector<TNumber> left, Tensors.Vector<TNumber> right, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Memory;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        var vectorCount = SimdVector.Count<TNumber>();
        var sums = new ConcurrentDictionary<long, ISimdVector<TNumber>>();
        var done = 0L;

        if (left.Length >= vectorCount)
        {
            done = LazyParallelExecutor.For(
                0,
                left.Length - vectorCount,
                ManagedTensorBackend.ParallelizationThreshold / 2,
                vectorCount,
                i =>
                {
                    var leftVector = leftMemoryBlock.UnsafeGetVectorUnchecked<TNumber>(i);
                    var rightVector = rightMemoryBlock.UnsafeGetVectorUnchecked<TNumber>(i);

                    _ = sums.AddOrUpdate(
                        i / vectorCount,
                        _ => leftVector.Multiply(rightVector),
                        (_, sum) => sum.Add(leftVector.Multiply(rightVector)));
                });
        }

        for (var i = done; i < left.Length; i++)
        {
            resultMemoryBlock[0] += leftMemoryBlock[i] * rightMemoryBlock[i];
        }

        if (sums.Values.Count > 0)
        {
            resultMemoryBlock[0] += sums.Values.Aggregate((x, y) => x.Add(y)).Sum();
        }
    }

    private static unsafe void MatrixMultiply8x8AvxFloatParallel(
        float* leftPtr,
        float* rightPtr,
        float* resultPtr,
        int leftRows,
        int rightCols,
        int leftCols)
    {
        // Define the window size
        // If you change this, ensure loops are unrolled accordingly
        const int blockDimSize = 8;

        // Calculate the largest multiples of 8 for each dimension
        var rowBlockEnd = leftRows / blockDimSize * blockDimSize;
        var colBlockEnd = rightCols / blockDimSize * blockDimSize;
        var depthBlockEnd = leftCols / blockDimSize * blockDimSize;

        LazyParallelExecutor.ForBlocked(
            0,
            rowBlockEnd,
            0,
            colBlockEnd,
            blockDimSize,
            blockDimSize,
            (rowBase, colBase) =>
            {
                // One accumulator for each of the 8 rows in the block
                var acc0 = Vector256<float>.Zero;
                var acc1 = Vector256<float>.Zero;
                var acc2 = Vector256<float>.Zero;
                var acc3 = Vector256<float>.Zero;
                var acc4 = Vector256<float>.Zero;
                var acc5 = Vector256<float>.Zero;
                var acc6 = Vector256<float>.Zero;
                var acc7 = Vector256<float>.Zero;

                // For each depth‐block of 8
                for (var depthBase = 0; depthBase < depthBlockEnd; depthBase += blockDimSize)
                {
                    // Unroll over the 8 depth positions inside this block
                    for (var depthOffset = 0; depthOffset < blockDimSize; ++depthOffset)
                    {
                        var rightRowPtr = rightPtr + (((depthBase + depthOffset) * rightCols) + colBase);
                        var rightRow = Avx.LoadVector256(rightRowPtr);

                        // row 0
                        var leftPtr0 = leftPtr + (((rowBase + 0) * leftCols) + (depthBase + depthOffset));
                        var leftVec0 = Avx.BroadcastScalarToVector256(leftPtr0);
                        acc0 = Avx.Add(acc0, Avx.Multiply(leftVec0, rightRow));

                        // row 1
                        var leftPtr1 = leftPtr + (((rowBase + 1) * leftCols) + (depthBase + depthOffset));
                        var leftVec1 = Avx.BroadcastScalarToVector256(leftPtr1);
                        acc1 = Avx.Add(acc1, Avx.Multiply(leftVec1, rightRow));

                        // row 2
                        var leftPtr2 = leftPtr + (((rowBase + 2) * leftCols) + (depthBase + depthOffset));
                        var leftVec2 = Avx.BroadcastScalarToVector256(leftPtr2);
                        acc2 = Avx.Add(acc2, Avx.Multiply(leftVec2, rightRow));

                        // row 3
                        var leftPtr3 = leftPtr + (((rowBase + 3) * leftCols) + (depthBase + depthOffset));
                        var leftVec3 = Avx.BroadcastScalarToVector256(leftPtr3);
                        acc3 = Avx.Add(acc3, Avx.Multiply(leftVec3, rightRow));

                        // row 4
                        var leftPtr4 = leftPtr + (((rowBase + 4) * leftCols) + (depthBase + depthOffset));
                        var leftVec4 = Avx.BroadcastScalarToVector256(leftPtr4);
                        acc4 = Avx.Add(acc4, Avx.Multiply(leftVec4, rightRow));

                        // row 5
                        var leftPtr5 = leftPtr + (((rowBase + 5) * leftCols) + (depthBase + depthOffset));
                        var leftVec5 = Avx.BroadcastScalarToVector256(leftPtr5);
                        acc5 = Avx.Add(acc5, Avx.Multiply(leftVec5, rightRow));

                        // row 6
                        var leftPtr6 = leftPtr + (((rowBase + 6) * leftCols) + (depthBase + depthOffset));
                        var leftVec6 = Avx.BroadcastScalarToVector256(leftPtr6);
                        acc6 = Avx.Add(acc6, Avx.Multiply(leftVec6, rightRow));

                        // row 7 of A
                        var leftPtr7 = leftPtr + (((rowBase + 7) * leftCols) + (depthBase + depthOffset));
                        var leftVec7 = Avx.BroadcastScalarToVector256(leftPtr7);
                        acc7 = Avx.Add(acc7, Avx.Multiply(leftVec7, rightRow));
                    }
                }

                // Store 8x8 accumulated results into resultPtr
                Avx.Store(resultPtr + (((rowBase + 0) * rightCols) + colBase), acc0);
                Avx.Store(resultPtr + (((rowBase + 1) * rightCols) + colBase), acc1);
                Avx.Store(resultPtr + (((rowBase + 2) * rightCols) + colBase), acc2);
                Avx.Store(resultPtr + (((rowBase + 3) * rightCols) + colBase), acc3);
                Avx.Store(resultPtr + (((rowBase + 4) * rightCols) + colBase), acc4);
                Avx.Store(resultPtr + (((rowBase + 5) * rightCols) + colBase), acc5);
                Avx.Store(resultPtr + (((rowBase + 6) * rightCols) + colBase), acc6);
                Avx.Store(resultPtr + (((rowBase + 7) * rightCols) + colBase), acc7);

                // Handle any tails for this tile
                for (var k = depthBlockEnd; k < leftCols; ++k)
                {
                    for (var rowOffset = 0; rowOffset < blockDimSize; ++rowOffset)
                    {
                        var leftVal = leftPtr[((rowBase + rowOffset) * leftCols) + k];
                        var resultRowPtr = resultPtr + (((rowBase + rowOffset) * rightCols) + colBase);
                        var rightRowPtr = rightPtr + ((k * rightCols) + colBase);

                        // Unrolled across 8 columns:
                        resultRowPtr[0] += leftVal * rightRowPtr[0];
                        resultRowPtr[1] += leftVal * rightRowPtr[1];
                        resultRowPtr[2] += leftVal * rightRowPtr[2];
                        resultRowPtr[3] += leftVal * rightRowPtr[3];
                        resultRowPtr[4] += leftVal * rightRowPtr[4];
                        resultRowPtr[5] += leftVal * rightRowPtr[5];
                        resultRowPtr[6] += leftVal * rightRowPtr[6];
                        resultRowPtr[7] += leftVal * rightRowPtr[7];
                    }
                }
            });

        // Handle tail columns
        if (colBlockEnd < rightCols)
        {
            for (var i = 0; i < rowBlockEnd; ++i)
            {
                for (var j = colBlockEnd; j < rightCols; ++j)
                {
                    var sum = 0f;
                    var aRowPtr = leftPtr + (i * leftCols);
                    var bColPtr = rightPtr + j;
                    for (var k = 0; k < leftCols; ++k)
                    {
                        sum += aRowPtr[k] * bColPtr[k * rightCols];
                    }

                    resultPtr[(i * rightCols) + j] = sum;
                }
            }
        }

        // Handle tail rows
        if (rowBlockEnd < leftRows)
        {
            for (var i = rowBlockEnd; i < leftRows; ++i)
            {
                for (var j = 0; j < colBlockEnd; ++j)
                {
                    var sum = 0f;
                    var aRowPtr = leftPtr + (i * leftCols);
                    var bColPtr = rightPtr + j;
                    for (var k = 0; k < leftCols; ++k)
                    {
                        sum += aRowPtr[k] * bColPtr[k * rightCols];
                    }

                    resultPtr[(i * rightCols) + j] = sum;
                }
            }
        }

        // Handle the bottom‐right corner
        if (rowBlockEnd < leftRows && colBlockEnd < rightCols)
        {
            for (var i = rowBlockEnd; i < leftRows; ++i)
            {
                for (var j = colBlockEnd; j < rightCols; ++j)
                {
                    var sum = 0f;
                    var aRowPtr = leftPtr + (i * leftCols);
                    var bColPtr = rightPtr + j;
                    for (var k = 0; k < leftCols; ++k)
                    {
                        sum += aRowPtr[k] * bColPtr[k * rightCols];
                    }

                    resultPtr[(i * rightCols) + j] = sum;
                }
            }
        }
    }

    private static unsafe void MatrixMultiply8X8AvxDoubleParallel(
        double* leftPtr,
        double* rightPtr,
        double* resultPtr,
        int leftRows,
        int rightCols,
        int leftCols)
    {
        // Define the window size
        // If you change this, ensure loops are unrolled accordingly
        const int blockDimSize = 8;

        var rowBlockEnd = leftRows / 8 * 8;
        var colBlockEnd = rightCols / 8 * 8;
        var depthBlockEnd = leftCols / 8 * 8;

        LazyParallelExecutor.ForBlocked(
            0,
            rowBlockEnd,
            0,
            colBlockEnd,
            blockDimSize,
            blockDimSize,
            (rowBase, colBase) =>
            {
                // Two accumulators for each of the 8 rows in the block
                var acc0Lo = Vector256<double>.Zero;
                var acc0Hi = Vector256<double>.Zero;
                var acc1Lo = Vector256<double>.Zero;
                var acc1Hi = Vector256<double>.Zero;
                var acc2Lo = Vector256<double>.Zero;
                var acc2Hi = Vector256<double>.Zero;
                var acc3Lo = Vector256<double>.Zero;
                var acc3Hi = Vector256<double>.Zero;
                var acc4Lo = Vector256<double>.Zero;
                var acc4Hi = Vector256<double>.Zero;
                var acc5Lo = Vector256<double>.Zero;
                var acc5Hi = Vector256<double>.Zero;
                var acc6Lo = Vector256<double>.Zero;
                var acc6Hi = Vector256<double>.Zero;
                var acc7Lo = Vector256<double>.Zero;
                var acc7Hi = Vector256<double>.Zero;

                // For each depth‐block of 8
                for (var depthBase = 0; depthBase < depthBlockEnd; depthBase += blockDimSize)
                {
                    // Unroll over the 8 depth positions inside this block
                    for (var depthOffset = 0; depthOffset < blockDimSize; ++depthOffset)
                    {
                        // Load the right row for this depth position:
                        var rightRowPtr = rightPtr + (((depthBase + depthOffset) * rightCols) + colBase);
                        var rightRowLo = Avx.LoadVector256(rightRowPtr);
                        var rightRowwHigh = Avx.LoadVector256(rightRowPtr + 4);

                        // Row 0
                        var leftPtr0 = leftPtr + (((rowBase + 0) * leftCols) + (depthBase + depthOffset));
                        var leftVec0 = Avx.BroadcastScalarToVector256(leftPtr0);
                        acc0Lo = Avx.Add(acc0Lo, Avx.Multiply(leftVec0, rightRowLo));
                        acc0Hi = Avx.Add(acc0Hi, Avx.Multiply(leftVec0, rightRowwHigh));

                        // Row 1
                        var leftPtr1 = leftPtr + (((rowBase + 1) * leftCols) + (depthBase + depthOffset));
                        var leftVec1 = Avx.BroadcastScalarToVector256(leftPtr1);
                        acc1Lo = Avx.Add(acc1Lo, Avx.Multiply(leftVec1, rightRowLo));
                        acc1Hi = Avx.Add(acc1Hi, Avx.Multiply(leftVec1, rightRowwHigh));

                        // Row 2
                        var leftPtr2 = leftPtr + (((rowBase + 2) * leftCols) + (depthBase + depthOffset));
                        var leftVec2 = Avx.BroadcastScalarToVector256(leftPtr2);
                        acc2Lo = Avx.Add(acc2Lo, Avx.Multiply(leftVec2, rightRowLo));
                        acc2Hi = Avx.Add(acc2Hi, Avx.Multiply(leftVec2, rightRowwHigh));

                        // Row 3
                        var leftPtr3 = leftPtr + (((rowBase + 3) * leftCols) + (depthBase + depthOffset));
                        var leftVec3 = Avx.BroadcastScalarToVector256(leftPtr3);
                        acc3Lo = Avx.Add(acc3Lo, Avx.Multiply(leftVec3, rightRowLo));
                        acc3Hi = Avx.Add(acc3Hi, Avx.Multiply(leftVec3, rightRowwHigh));

                        // Row 4
                        var leftPtr4 = leftPtr + (((rowBase + 4) * leftCols) + (depthBase + depthOffset));
                        var leftVec4 = Avx.BroadcastScalarToVector256(leftPtr4);
                        acc4Lo = Avx.Add(acc4Lo, Avx.Multiply(leftVec4, rightRowLo));
                        acc4Hi = Avx.Add(acc4Hi, Avx.Multiply(leftVec4, rightRowwHigh));

                        // Row 5
                        var leftPtr5 = leftPtr + (((rowBase + 5) * leftCols) + (depthBase + depthOffset));
                        var leftVec5 = Avx.BroadcastScalarToVector256(leftPtr5);
                        acc5Lo = Avx.Add(acc5Lo, Avx.Multiply(leftVec5, rightRowLo));
                        acc5Hi = Avx.Add(acc5Hi, Avx.Multiply(leftVec5, rightRowwHigh));

                        // Row 6
                        var leftPtr6 = leftPtr + (((rowBase + 6) * leftCols) + (depthBase + depthOffset));
                        var leftVec6 = Avx.BroadcastScalarToVector256(leftPtr6);
                        acc6Lo = Avx.Add(acc6Lo, Avx.Multiply(leftVec6, rightRowLo));
                        acc6Hi = Avx.Add(acc6Hi, Avx.Multiply(leftVec6, rightRowwHigh));

                        // Row 7
                        var leftPtr7 = leftPtr + (((rowBase + 7) * leftCols) + (depthBase + depthOffset));
                        var leftVec7 = Avx.BroadcastScalarToVector256(leftPtr7);
                        acc7Lo = Avx.Add(acc7Lo, Avx.Multiply(leftVec7, rightRowLo));
                        acc7Hi = Avx.Add(acc7Hi, Avx.Multiply(leftVec7, rightRowwHigh));
                    }
                }

                // Store 8x8 accumulated results into resultPtr
                var resultRow0 = resultPtr + (((rowBase + 0) * rightCols) + colBase);
                Avx.Store(resultRow0 + 0, acc0Lo);
                Avx.Store(resultRow0 + 4, acc0Hi);

                var resultRow1 = resultPtr + (((rowBase + 1) * rightCols) + colBase);
                Avx.Store(resultRow1 + 0, acc1Lo);
                Avx.Store(resultRow1 + 4, acc1Hi);

                var resultRow2 = resultPtr + (((rowBase + 2) * rightCols) + colBase);
                Avx.Store(resultRow2 + 0, acc2Lo);
                Avx.Store(resultRow2 + 4, acc2Hi);

                var resultRow3 = resultPtr + (((rowBase + 3) * rightCols) + colBase);
                Avx.Store(resultRow3 + 0, acc3Lo);
                Avx.Store(resultRow3 + 4, acc3Hi);

                var resultRow4 = resultPtr + (((rowBase + 4) * rightCols) + colBase);
                Avx.Store(resultRow4 + 0, acc4Lo);
                Avx.Store(resultRow4 + 4, acc4Hi);

                var resultRow5 = resultPtr + (((rowBase + 5) * rightCols) + colBase);
                Avx.Store(resultRow5 + 0, acc5Lo);
                Avx.Store(resultRow5 + 4, acc5Hi);

                var resultRow6 = resultPtr + (((rowBase + 6) * rightCols) + colBase);
                Avx.Store(resultRow6 + 0, acc6Lo);
                Avx.Store(resultRow6 + 4, acc6Hi);

                var resultRow7 = resultPtr + (((rowBase + 7) * rightCols) + colBase);
                Avx.Store(resultRow7 + 0, acc7Lo);
                Avx.Store(resultRow7 + 4, acc7Hi);

                // Handle any tails for this tile
                for (var k = depthBlockEnd; k < leftCols; ++k)
                {
                    for (var rowOffset = 0; rowOffset < 8; ++rowOffset)
                    {
                        var aVal = leftPtr[((rowBase + rowOffset) * leftCols) + k];
                        var cRowPtr = resultPtr + (((rowBase + rowOffset) * rightCols) + colBase);
                        var bRowPtr = rightPtr + ((k * rightCols) + colBase);

                        // Unrolled across 8 columns:
                        cRowPtr[0] += aVal * bRowPtr[0];
                        cRowPtr[1] += aVal * bRowPtr[1];
                        cRowPtr[2] += aVal * bRowPtr[2];
                        cRowPtr[3] += aVal * bRowPtr[3];
                        cRowPtr[4] += aVal * bRowPtr[4];
                        cRowPtr[5] += aVal * bRowPtr[5];
                        cRowPtr[6] += aVal * bRowPtr[6];
                        cRowPtr[7] += aVal * bRowPtr[7];
                    }
                }
            });

        // Handle tail columns
        if (colBlockEnd < rightCols)
        {
            for (var i = 0; i < rowBlockEnd; ++i)
            {
                for (var j = colBlockEnd; j < rightCols; ++j)
                {
                    var sum = 0.0;
                    var aRowPtr = leftPtr + (i * leftCols);
                    var bColPtr = rightPtr + j;
                    for (var k = 0; k < leftCols; ++k)
                    {
                        sum += aRowPtr[k] * bColPtr[k * rightCols];
                    }

                    resultPtr[(i * rightCols) + j] = sum;
                }
            }
        }

        // Handle tail rows
        if (rowBlockEnd < leftRows)
        {
            for (var i = rowBlockEnd; i < leftRows; ++i)
            {
                for (var j = 0; j < colBlockEnd; ++j)
                {
                    var sum = 0.0;
                    var aRowPtr = leftPtr + (i * leftCols);
                    var bColPtr = rightPtr + j;
                    for (var k = 0; k < leftCols; ++k)
                    {
                        sum += aRowPtr[k] * bColPtr[k * rightCols];
                    }

                    resultPtr[(i * rightCols) + j] = sum;
                }
            }
        }

        // Handle the bottom‐right corner
        if (rowBlockEnd < leftRows && colBlockEnd < rightCols)
        {
            for (var i = rowBlockEnd; i < leftRows; ++i)
            {
                for (var j = colBlockEnd; j < rightCols; ++j)
                {
                    var sum = 0.0;
                    var aRowPtr = leftPtr + (i * leftCols);
                    var bColPtr = rightPtr + j;
                    for (var k = 0; k < leftCols; ++k)
                    {
                        sum += aRowPtr[k] * bColPtr[k * rightCols];
                    }

                    resultPtr[(i * rightCols) + j] = sum;
                }
            }
        }
    }
}