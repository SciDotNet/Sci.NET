// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Intrinsics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Backends.Managed.Buffers;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedLinearAlgebraKernels : ILinearAlgebraKernels
{
    private const int GemmMrFp32 = 8;
    private const int GemmNrFp32 = 8;
    private const int GemmMrFp64 = 4;
    private const int GemmNrFp64 = 4;
    private const int GemmMcFp32 = 256;
    private const int GemmKcFp32 = 128;
    private const int GemmNcFp32 = 256;
    private const int GemmMcFp64 = 128;
    private const int GemmKcFp64 = 128;
    private const int GemmNcFp64 = 128;

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
        const int iBlock = 128; // General block size for rows
        const int jBlock = 16; // General block size for columns

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            (IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Fma) != 0 &&
            typeof(TNumber) == typeof(float))
        {
            GemmFp32(
                (float*)leftMemoryBlockPtr,
                (float*)rightMemoryBlockPtr,
                (float*)resultMemoryBlockPtr,
                leftRows,
                rightColumns,
                leftColumns);
            return;
        }

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            (IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Fma) != 0 &&
            typeof(TNumber) == typeof(double))
        {
            GemmFp64(
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
                        for (var k = 0; k < leftColumns; ++k)
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

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            (IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Fma) != 0 &&
            typeof(TNumber) == typeof(float))
        {
            InnerProductFp32Avx(
                (float*)leftMemoryBlock.Pointer,
                (float*)rightMemoryBlock.Pointer,
                (float*)resultMemoryBlock.Pointer,
                left.Length);

            return;
        }

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            (IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Fma) != 0 &&
            typeof(TNumber) == typeof(double))
        {
            InnerProductFp64Avx(
                (double*)leftMemoryBlock.Pointer,
                (double*)rightMemoryBlock.Pointer,
                (double*)resultMemoryBlock.Pointer,
                left.Length);

            return;
        }

        using var sums = new ThreadLocal<TNumber>(() => TNumber.Zero, true);

        _ = LazyParallelExecutor.For(
            0,
            left.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var leftVector = leftMemoryBlock[i];
                var rightVector = rightMemoryBlock[i];
                sums.Value += leftVector * rightVector;
            });

        var sum = TNumber.Zero;
        foreach (var threadSum in sums.Values)
        {
            sum += threadSum;
        }

        resultMemoryBlock[0] = sum;
    }

    private static unsafe void GemmFp32(
        float* a,
        float* b,
        float* c,
        int m,
        int n,
        int k)
    {
        var numTiles = (m + GemmMcFp32 - 1) / GemmMcFp32;

        _ = Parallel.For(
            0,
            numTiles,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () =>
            {
                var aBuffer = (float*)NativeMemory.AlignedAlloc(GemmMcFp32 * GemmKcFp32 * sizeof(float), 64);
                var bBuffer = (float*)NativeMemory.AlignedAlloc(GemmKcFp32 * GemmNcFp32 * sizeof(float), 64);
                return new Panel2dFp32(aBuffer, bBuffer);
            },
            (tileIdx, _, panels) =>
            {
                var mBase = tileIdx * GemmMcFp32;
                var mTile = Math.Min(GemmMcFp32, m - mBase);
                var aPanel = panels.A;
                var bPanel = panels.B;

                for (var kBase = 0; kBase < k; kBase += GemmKcFp32)
                {
                    var kTile = Math.Min(GemmKcFp32, k - kBase);
                    PackAFp32(a + (mBase * k) + kBase, aPanel, k, mTile, kTile);

                    for (var nBase = 0; nBase < n; nBase += GemmNcFp32)
                    {
                        var nc = Math.Min(GemmNcFp32, n - nBase);
                        PackBFp32(b + (kBase * n) + nBase, bPanel, n, kTile, nc);

                        for (var mReg = 0; mReg < mTile; mReg += GemmMrFp32)
                        {
                            for (var nReg = 0; nReg < nc; nReg += GemmNrFp32)
                            {
                                MicroKernel8x8Fp32(
                                    aPanel + (mReg * GemmKcFp32),
                                    bPanel + nReg,
                                    c + ((mBase + mReg) * n) + nBase + nReg,
                                    kTile,
                                    n);
                            }
                        }
                    }
                }

                return panels;
            },
            panels =>
            {
                NativeMemory.AlignedFree(panels.A);
                NativeMemory.AlignedFree(panels.B);
            });
    }

    private static unsafe void GemmFp64(
        double* a,
        double* b,
        double* c,
        int m,
        int n,
        int k)
    {
        var numTiles = (m + GemmMcFp64 - 1) / GemmMcFp64;

        _ = Parallel.For(
            0,
            numTiles,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () =>
            {
                var aBuffer = (double*)NativeMemory.AlignedAlloc(GemmMcFp64 * GemmKcFp64 * sizeof(double), 64);
                var bBuffer = (double*)NativeMemory.AlignedAlloc(GemmKcFp64 * GemmNcFp64 * sizeof(double), 64);
                return new Panel2dFp64(aBuffer, bBuffer);
            },
            (tileIdx, _, panels) =>
            {
                var mBase = tileIdx * GemmMcFp64;
                var mTile = Math.Min(GemmMcFp64, m - mBase);
                var aPanel = panels.A;
                var bPanel = panels.B;

                for (var kBase = 0; kBase < k; kBase += GemmKcFp64)
                {
                    var kTile = Math.Min(GemmKcFp64, k - kBase);
                    PackAFp64(a + (mBase * k) + kBase, aPanel, k, mTile, kTile);

                    for (var nBase = 0; nBase < n; nBase += GemmNcFp64)
                    {
                        var nc = Math.Min(GemmNcFp64, n - nBase);
                        PackBFp64(b + (kBase * n) + nBase, bPanel, n, kTile, nc);

                        for (var mReg = 0; mReg < mTile; mReg += GemmMrFp64)
                        {
                            for (var nReg = 0; nReg < nc; nReg += GemmNrFp64)
                            {
                                MicroKernel4x4Fp64(
                                    aPanel + (mReg * GemmKcFp64),
                                    bPanel + nReg,
                                    c + ((mBase + mReg) * n) + nBase + nReg,
                                    kTile,
                                    n);
                            }
                        }
                    }
                }

                return panels;
            },
            panels =>
            {
                NativeMemory.AlignedFree(panels.A);
                NativeMemory.AlignedFree(panels.B);
            });
    }

    private static unsafe void MicroKernel8x8Fp32(
        float* aPanel,
        float* bPanel,
        float* cTile,
        int kTile,
        int ldc)
    {
        var c0 = Avx.LoadVector256(cTile + (0 * ldc));
        var c1 = Avx.LoadVector256(cTile + (1 * ldc));
        var c2 = Avx.LoadVector256(cTile + (2 * ldc));
        var c3 = Avx.LoadVector256(cTile + (3 * ldc));
        var c4 = Avx.LoadVector256(cTile + (4 * ldc));
        var c5 = Avx.LoadVector256(cTile + (5 * ldc));
        var c6 = Avx.LoadVector256(cTile + (6 * ldc));
        var c7 = Avx.LoadVector256(cTile + (7 * ldc));

        var a0 = aPanel + (0 * GemmKcFp32);
        var a1 = aPanel + (1 * GemmKcFp32);
        var a2 = aPanel + (2 * GemmKcFp32);
        var a3 = aPanel + (3 * GemmKcFp32);
        var a4 = aPanel + (4 * GemmKcFp32);
        var a5 = aPanel + (5 * GemmKcFp32);
        var a6 = aPanel + (6 * GemmKcFp32);
        var a7 = aPanel + (7 * GemmKcFp32);

        for (var k = 0; k < kTile; k++)
        {
            var bVec = Avx.LoadVector256(bPanel + (k * GemmNcFp32));

            c0 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(a0 + k), bVec, c0);
            c1 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(a1 + k), bVec, c1);
            c2 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(a2 + k), bVec, c2);
            c3 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(a3 + k), bVec, c3);
            c4 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(a4 + k), bVec, c4);
            c5 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(a5 + k), bVec, c5);
            c6 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(a6 + k), bVec, c6);
            c7 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(a7 + k), bVec, c7);
        }

        Avx.Store(cTile + (0 * ldc), c0);
        Avx.Store(cTile + (1 * ldc), c1);
        Avx.Store(cTile + (2 * ldc), c2);
        Avx.Store(cTile + (3 * ldc), c3);
        Avx.Store(cTile + (4 * ldc), c4);
        Avx.Store(cTile + (5 * ldc), c5);
        Avx.Store(cTile + (6 * ldc), c6);
        Avx.Store(cTile + (7 * ldc), c7);
    }

    private static unsafe void MicroKernel4x4Fp64(
        double* aPanel,
        double* bPanel,
        double* cTile,
        int kTile,
        int ldc)
    {
        var c0 = Avx.LoadVector256(cTile + (0 * ldc));
        var c1 = Avx.LoadVector256(cTile + (1 * ldc));
        var c2 = Avx.LoadVector256(cTile + (2 * ldc));
        var c3 = Avx.LoadVector256(cTile + (3 * ldc));

        var a0 = aPanel + (0 * GemmKcFp64);
        var a1 = aPanel + (1 * GemmKcFp64);
        var a2 = aPanel + (2 * GemmKcFp64);
        var a3 = aPanel + (3 * GemmKcFp64);

        for (var k = 0; k < kTile; k++)
        {
            var bVec = Avx.LoadVector256(bPanel + (k * GemmNcFp64));

            c0 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(a0 + k), bVec, c0);
            c1 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(a1 + k), bVec, c1);
            c2 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(a2 + k), bVec, c2);
            c3 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(a3 + k), bVec, c3);
        }

        Avx.Store(cTile + (0 * ldc), c0);
        Avx.Store(cTile + (1 * ldc), c1);
        Avx.Store(cTile + (2 * ldc), c2);
        Avx.Store(cTile + (3 * ldc), c3);
    }

    private static unsafe void InnerProductFp32Avx(float* leftMemoryPtr, float* rightMemoryPtr, float* resultPtr, long n)
    {
        if (n <= 0)
        {
            resultPtr[0] = 0.0f;
            return;
        }

        var processes = Environment.ProcessorCount;
        var partials = new float[processes];

        _ = Parallel.For(
            0,
            processes,
            tid =>
            {
                var start = tid * n / processes;
                var end = (tid + 1) * n / processes;

                var i = start;
                var acc0 = Vector256<float>.Zero;
                var acc1 = Vector256<float>.Zero;
                var acc2 = Vector256<float>.Zero;
                var acc3 = Vector256<float>.Zero;

                for (; i + 32 <= end; i += 32)
                {
                    var vLeft0 = Avx.LoadVector256(leftMemoryPtr + i + 0);
                    var vRight0 = Avx.LoadVector256(rightMemoryPtr + i + 0);
                    var vLeft1 = Avx.LoadVector256(leftMemoryPtr + i + 8);
                    var vRight1 = Avx.LoadVector256(rightMemoryPtr + i + 8);
                    var vLeft2 = Avx.LoadVector256(leftMemoryPtr + i + 16);
                    var vRight2 = Avx.LoadVector256(rightMemoryPtr + i + 16);
                    var vLeft3 = Avx.LoadVector256(leftMemoryPtr + i + 24);
                    var vRight3 = Avx.LoadVector256(rightMemoryPtr + i + 24);

                    if (Fma.IsSupported)
                    {
                        acc0 = Fma.MultiplyAdd(vLeft0, vRight0, acc0);
                        acc1 = Fma.MultiplyAdd(vLeft1, vRight1, acc1);
                        acc2 = Fma.MultiplyAdd(vLeft2, vRight2, acc2);
                        acc3 = Fma.MultiplyAdd(vLeft3, vRight3, acc3);
                    }
                    else
                    {
                        acc0 = Avx.Add(acc0, Avx.Multiply(vLeft0, vRight0));
                        acc1 = Avx.Add(acc1, Avx.Multiply(vLeft1, vRight1));
                        acc2 = Avx.Add(acc2, Avx.Multiply(vLeft2, vRight2));
                        acc3 = Avx.Add(acc3, Avx.Multiply(vLeft3, vRight3));
                    }
                }

                for (; i + 8 <= end; i += 8)
                {
                    var va = Avx.LoadVector256(leftMemoryPtr + i);
                    var vb = Avx.LoadVector256(rightMemoryPtr + i);
                    acc0 = Fma.IsSupported ? Fma.MultiplyAdd(va, vb, acc0) : Avx.Add(acc0, Avx.Multiply(va, vb));
                }

                var acc = Avx.Add(Avx.Add(acc0, acc1), Avx.Add(acc2, acc3));
                var tmp = Avx.HorizontalAdd(acc, acc);
                tmp = Avx.HorizontalAdd(tmp, tmp);

                var sumVec = Avx.Add(tmp, Avx.Permute2x128(tmp, tmp, 0x01));
                var sum = sumVec.GetElement(0);

                for (; i < end; ++i)
                {
                    sum += leftMemoryPtr[i] * rightMemoryPtr[i];
                }

                partials[tid] = sum;
            });

        // Neumaier Sum for accuracy
        float s = 0, c = 0;
        for (var i = 0; i < partials.Length; i++)
        {
            var t = s + partials[i];
            c += float.Abs(s) >= float.Abs(partials[i]) ? s - t + partials[i] : partials[i] - t + s;
            s = t;
        }

        resultPtr[0] = s + c;
    }

    private static unsafe void InnerProductFp64Avx(double* leftMemoryPtr, double* rightMemoryPtr, double* resultPtr, long n)
    {
        if (n <= 0)
        {
            resultPtr[0] = 0.0d;
            return;
        }

        var processes = Environment.ProcessorCount;
        var partials = new double[processes];

        _ = Parallel.For(
            0,
            processes,
            tid =>
            {
                var start = tid * n / processes;
                var end = (tid + 1) * n / processes;

                var i = start;
                var acc0 = Vector256<double>.Zero;
                var acc1 = Vector256<double>.Zero;
                for (; i + 8 <= end; i += 8)
                {
                    var vLeft0 = Avx.LoadVector256(leftMemoryPtr + i + 0);
                    var vRight0 = Avx.LoadVector256(rightMemoryPtr + i + 0);
                    var vLeft1 = Avx.LoadVector256(leftMemoryPtr + i + 4);
                    var vRight1 = Avx.LoadVector256(rightMemoryPtr + i + 4);

                    if (Fma.IsSupported)
                    {
                        acc0 = Fma.MultiplyAdd(vLeft0, vRight0, acc0);
                        acc1 = Fma.MultiplyAdd(vLeft1, vRight1, acc1);
                    }
                    else
                    {
                        acc0 = Avx.Add(acc0, Avx.Multiply(vLeft0, vRight0));
                        acc1 = Avx.Add(acc1, Avx.Multiply(vLeft1, vRight1));
                    }
                }

                for (; i + 4 <= end; i += 4)
                {
                    var va = Avx.LoadVector256(leftMemoryPtr + i);
                    var vb = Avx.LoadVector256(rightMemoryPtr + i);
                    acc0 = Fma.IsSupported ? Fma.MultiplyAdd(va, vb, acc0) : Avx.Add(acc0, Avx.Multiply(va, vb));
                }

                var buf = stackalloc double[4];
                var acc = Avx.Add(acc0, acc1);
                Avx.Store(buf, acc);
                var sum = buf[0] + buf[1] + buf[2] + buf[3];

                for (; i < end; ++i)
                {
                    sum += leftMemoryPtr[i] * rightMemoryPtr[i];
                }

                partials[tid] = sum;
            });

        // Neumaier Sum for accuracy
        double s = 0, c = 0;
        for (var i = 0; i < partials.Length; i++)
        {
            var t = s + partials[i];
            c += double.Abs(s) >= double.Abs(partials[i]) ? s - t + partials[i] : partials[i] - t + s;
            s = t;
        }

        resultPtr[0] = s + c;
    }

    private static unsafe void PackAFp32(float* src, float* dstPanel, int lda, int mTile, int kTile)
    {
        for (var i = 0; i < GemmMcFp32; i++)
        {
            for (var k = 0; k < GemmKcFp32; k++)
            {
                dstPanel[(i * GemmKcFp32) + k] = i < mTile && k < kTile ? src[(i * lda) + k] : 0f;
            }
        }
    }

    private static unsafe void PackBFp32(float* src, float* dstPanel, int ldb, int kTile, int nTile)
    {
        for (var k = 0; k < GemmKcFp32; k++)
        {
            for (var j = 0; j < GemmNcFp32; j++)
            {
                dstPanel[(k * GemmNcFp32) + j] =
                    k < kTile && j < nTile ? src[(k * ldb) + j] : 0f;
            }
        }
    }

    private static unsafe void PackAFp64(
        double* src,
        double* dstPanel,
        int lda,
        int mTile,
        int kTile)
    {
        for (var i = 0; i < GemmMcFp64; i++)
        {
            for (var k = 0; k < GemmKcFp64; k++)
            {
                dstPanel[(i * GemmKcFp64) + k] = i < mTile && k < kTile ? src[(i * lda) + k] : 0d;
            }
        }
    }

    private static unsafe void PackBFp64(
        double* src,
        double* dstPanel,
        int ldb,
        int kTile,
        int nTile)
    {
        for (var k = 0; k < GemmKcFp64; k++)
        {
            for (var j = 0; j < GemmNcFp64; j++)
            {
                dstPanel[(k * GemmNcFp64) + j] = k < kTile && j < nTile ? src[(k * ldb) + j] : 0d;
            }
        }
    }
}