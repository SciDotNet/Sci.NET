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

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedEqualityOperationKernels : IEqualityOperationKernels
{
    public unsafe void PointwiseEqualsKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(float))
        {
            PointwiseEqualsAvxFp32(
                (float*)leftBlock.Pointer,
                (float*)rightBlock.Pointer,
                (float*)resultBlock.Pointer,
                n);
            return;
        }

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(double))
        {
            PointwiseEqualsAvxFp64(
                (double*)leftBlock.Pointer,
                (double*)rightBlock.Pointer,
                (double*)resultBlock.Pointer,
                n);
            return;
        }

        _ = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = leftBlock[i] == rightBlock[i] ? TNumber.One : TNumber.Zero);
    }

    public unsafe void PointwiseNotEqualKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(float))
        {
            PointwiseNotEqualsAvxFp32(
                (float*)leftBlock.Pointer,
                (float*)rightBlock.Pointer,
                (float*)resultBlock.Pointer,
                n);
            return;
        }

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(double))
        {
            PointwiseNotEqualsAvxFp64(
                (double*)leftBlock.Pointer,
                (double*)rightBlock.Pointer,
                (double*)resultBlock.Pointer,
                n);
            return;
        }

        _ = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = leftBlock[i] == rightBlock[i] ? TNumber.Zero : TNumber.One);
    }

    public unsafe void PointwiseGreaterThanKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(float))
        {
            PointwiseGreaterThanAvxFp32(
                (float*)leftBlock.Pointer,
                (float*)rightBlock.Pointer,
                (float*)resultBlock.Pointer,
                n);
            return;
        }

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(double))
        {
            PointwiseGreaterThanAvxFp64(
                (double*)leftBlock.Pointer,
                (double*)rightBlock.Pointer,
                (double*)resultBlock.Pointer,
                n);
            return;
        }

        _ = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = leftBlock[i] > rightBlock[i] ? TNumber.One : TNumber.Zero);
    }

    public unsafe void PointwiseGreaterThanOrEqualKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(float))
        {
            PointwiseGreaterThanOrEqualAvxFp32(
                (float*)leftBlock.Pointer,
                (float*)rightBlock.Pointer,
                (float*)resultBlock.Pointer,
                n);
            return;
        }

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(double))
        {
            PointwiseGreaterThanOrEqualAvxFp64(
                (double*)leftBlock.Pointer,
                (double*)rightBlock.Pointer,
                (double*)resultBlock.Pointer,
                n);
            return;
        }

        _ = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = leftBlock[i] >= rightBlock[i] ? TNumber.One : TNumber.Zero);
    }

    public unsafe void PointwiseLessThanKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(float))
        {
            PointwiseLessThanAvxFp32(
                (float*)leftBlock.Pointer,
                (float*)rightBlock.Pointer,
                (float*)resultBlock.Pointer,
                n);
            return;
        }

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(double))
        {
            PointwiseLessThanAvxFp64(
                (double*)leftBlock.Pointer,
                (double*)rightBlock.Pointer,
                (double*)resultBlock.Pointer,
                n);
            return;
        }

        _ = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = leftBlock[i] < rightBlock[i] ? TNumber.One : TNumber.Zero);
    }

    public unsafe void PointwiseLessThanOrEqualKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(float))
        {
            PointwiseLessThanOrEqualAvxFp32(
                (float*)leftBlock.Pointer,
                (float*)rightBlock.Pointer,
                (float*)resultBlock.Pointer,
                n);
            return;
        }

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(double))
        {
            PointwiseLessThanOrEqualAvxFp64(
                (double*)leftBlock.Pointer,
                (double*)rightBlock.Pointer,
                (double*)resultBlock.Pointer,
                n);
            return;
        }

        _ = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = leftBlock[i] <= rightBlock[i] ? TNumber.One : TNumber.Zero);
    }

    private static unsafe void PointwiseEqualsAvxFp32(
        float* leftMemoryPointer,
        float* rightMemoryPointer,
        float* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp32 - 1) / NativeBufferHelpers.TileSizeFp32;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () =>
            {
                var a = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                var b = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                return new Panel2dFp32(a, b);
            },
            (tileIdx, _, panel) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp32;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp32, n);
                var count = tileEnd - tileStart;
                var one = Vector256<float>.One;
                var zero = Vector256<float>.Zero;

                NativeBufferHelpers.Pack1dFp32Avx(leftMemoryPointer + tileStart, panel.A, count);
                NativeBufferHelpers.Pack1dFp32Avx(rightMemoryPointer + tileStart, panel.B, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
                {
                    var leftVector = Avx.LoadVector256(panel.A + i);
                    var rightVector = Avx.LoadVector256(panel.B + i);
                    var equalsMask = Avx.CompareEqual(leftVector, rightVector);
                    var result = Avx.BlendVariable(zero, one, equalsMask);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = panel.A[i].Equals(panel.B[i]) ? 1f : 0f;
                }

                return panel;
            },
            data =>
            {
                NativeMemory.AlignedFree(data.A);
                NativeMemory.AlignedFree(data.B);
            });
    }

    private static unsafe void PointwiseEqualsAvxFp64(
        double* leftMemoryPointer,
        double* rightMemoryPointer,
        double* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp64 - 1) / NativeBufferHelpers.TileSizeFp64;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () =>
            {
                var a = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                var b = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                return new Panel2dFp64(a, b);
            },
            (tileIdx, _, panel) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp64;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp64, n);
                var count = tileEnd - tileStart;
                var one = Vector256<double>.One;
                var zero = Vector256<double>.Zero;

                NativeBufferHelpers.Pack1dFp64Avx(leftMemoryPointer + tileStart, panel.A, count);
                NativeBufferHelpers.Pack1dFp64Avx(rightMemoryPointer + tileStart, panel.B, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
                {
                    var leftVector = Avx.LoadVector256(panel.A + i);
                    var rightVector = Avx.LoadVector256(panel.B + i);
                    var equalsMask = Avx.CompareEqual(leftVector, rightVector);
                    var result = Avx.BlendVariable(zero, one, equalsMask);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = panel.A[i].Equals(panel.B[i]) ? 1f : 0f;
                }

                return panel;
            },
            data =>
            {
                NativeMemory.AlignedFree(data.A);
                NativeMemory.AlignedFree(data.B);
            });
    }

    private static unsafe void PointwiseNotEqualsAvxFp32(
        float* leftMemoryPointer,
        float* rightMemoryPointer,
        float* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp32 - 1) / NativeBufferHelpers.TileSizeFp32;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () =>
            {
                var a = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                var b = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                return new Panel2dFp32(a, b);
            },
            (tileIdx, _, panel) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp32;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp32, n);
                var count = tileEnd - tileStart;
                var one = Vector256<float>.One;
                var zero = Vector256<float>.Zero;

                NativeBufferHelpers.Pack1dFp32Avx(leftMemoryPointer + tileStart, panel.A, count);
                NativeBufferHelpers.Pack1dFp32Avx(rightMemoryPointer + tileStart, panel.B, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
                {
                    var leftVector = Avx.LoadVector256(panel.A + i);
                    var rightVector = Avx.LoadVector256(panel.B + i);
                    var equalsMask = Avx.CompareNotEqual(leftVector, rightVector);
                    var result = Avx.BlendVariable(zero, one, equalsMask);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = panel.A[i].Equals(panel.B[i]) ? 0f : 1f;
                }

                return panel;
            },
            data =>
            {
                NativeMemory.AlignedFree(data.A);
                NativeMemory.AlignedFree(data.B);
            });
    }

    private static unsafe void PointwiseNotEqualsAvxFp64(
        double* leftMemoryPointer,
        double* rightMemoryPointer,
        double* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp64 - 1) / NativeBufferHelpers.TileSizeFp64;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () =>
            {
                var a = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                var b = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                return new Panel2dFp64(a, b);
            },
            (tileIdx, _, panel) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp64;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp64, n);
                var count = tileEnd - tileStart;
                var one = Vector256<double>.One;
                var zero = Vector256<double>.Zero;

                NativeBufferHelpers.Pack1dFp64Avx(leftMemoryPointer + tileStart, panel.A, count);
                NativeBufferHelpers.Pack1dFp64Avx(rightMemoryPointer + tileStart, panel.B, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
                {
                    var leftVector = Avx.LoadVector256(panel.A + i);
                    var rightVector = Avx.LoadVector256(panel.B + i);
                    var equalsMask = Avx.CompareNotEqual(leftVector, rightVector);
                    var result = Avx.BlendVariable(zero, one, equalsMask);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = panel.A[i].Equals(panel.B[i]) ? 0f : 1f;
                }

                return panel;
            },
            data =>
            {
                NativeMemory.AlignedFree(data.A);
                NativeMemory.AlignedFree(data.B);
            });
    }

    private static unsafe void PointwiseGreaterThanAvxFp32(
        float* leftMemoryPointer,
        float* rightMemoryPointer,
        float* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp32 - 1) / NativeBufferHelpers.TileSizeFp32;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () =>
            {
                var a = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                var b = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                return new Panel2dFp32(a, b);
            },
            (tileIdx, _, panel) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp32;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp32, n);
                var count = tileEnd - tileStart;
                var one = Vector256<float>.One;
                var zero = Vector256<float>.Zero;

                NativeBufferHelpers.Pack1dFp32Avx(leftMemoryPointer + tileStart, panel.A, count);
                NativeBufferHelpers.Pack1dFp32Avx(rightMemoryPointer + tileStart, panel.B, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
                {
                    var leftVector = Avx.LoadVector256(panel.A + i);
                    var rightVector = Avx.LoadVector256(panel.B + i);
                    var greaterThanMask = Avx.CompareGreaterThan(leftVector, rightVector);
                    var result = Avx.BlendVariable(zero, one, greaterThanMask);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = panel.A[i] > panel.B[i] ? 1f : 0f;
                }

                return panel;
            },
            data =>
            {
                NativeMemory.AlignedFree(data.A);
                NativeMemory.AlignedFree(data.B);
            });
    }

    private static unsafe void PointwiseGreaterThanAvxFp64(
        double* leftMemoryPointer,
        double* rightMemoryPointer,
        double* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp64 - 1) / NativeBufferHelpers.TileSizeFp64;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () =>
            {
                var a = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                var b = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                return new Panel2dFp64(a, b);
            },
            (tileIdx, _, panel) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp64;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp64, n);
                var count = tileEnd - tileStart;
                var one = Vector256<double>.One;
                var zero = Vector256<double>.Zero;

                NativeBufferHelpers.Pack1dFp64Avx(leftMemoryPointer + tileStart, panel.A, count);
                NativeBufferHelpers.Pack1dFp64Avx(rightMemoryPointer + tileStart, panel.B, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
                {
                    var leftVector = Avx.LoadVector256(panel.A + i);
                    var rightVector = Avx.LoadVector256(panel.B + i);
                    var greaterThanMask = Avx.CompareGreaterThan(leftVector, rightVector);
                    var result = Avx.BlendVariable(zero, one, greaterThanMask);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = panel.A[i] > panel.B[i] ? 1f : 0f;
                }

                return panel;
            },
            data =>
            {
                NativeMemory.AlignedFree(data.A);
                NativeMemory.AlignedFree(data.B);
            });
    }

    private static unsafe void PointwiseGreaterThanOrEqualAvxFp32(
        float* leftMemoryPointer,
        float* rightMemoryPointer,
        float* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp32 - 1) / NativeBufferHelpers.TileSizeFp32;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () =>
            {
                var a = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                var b = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                return new Panel2dFp32(a, b);
            },
            (tileIdx, _, panel) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp32;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp32, n);
                var count = tileEnd - tileStart;
                var one = Vector256<float>.One;
                var zero = Vector256<float>.Zero;

                NativeBufferHelpers.Pack1dFp32Avx(leftMemoryPointer + tileStart, panel.A, count);
                NativeBufferHelpers.Pack1dFp32Avx(rightMemoryPointer + tileStart, panel.B, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
                {
                    var leftVector = Avx.LoadVector256(panel.A + i);
                    var rightVector = Avx.LoadVector256(panel.B + i);
                    var greaterThanOrEqualMask = Avx.CompareGreaterThanOrEqual(leftVector, rightVector);
                    var result = Avx.BlendVariable(zero, one, greaterThanOrEqualMask);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = panel.A[i] >= panel.B[i] ? 1f : 0f;
                }

                return panel;
            },
            data =>
            {
                NativeMemory.AlignedFree(data.A);
                NativeMemory.AlignedFree(data.B);
            });
    }

    private static unsafe void PointwiseGreaterThanOrEqualAvxFp64(
        double* leftMemoryPointer,
        double* rightMemoryPointer,
        double* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp64 - 1) / NativeBufferHelpers.TileSizeFp64;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () =>
            {
                var a = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                var b = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                return new Panel2dFp64(a, b);
            },
            (tileIdx, _, panel) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp64;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp64, n);
                var count = tileEnd - tileStart;
                var one = Vector256<double>.One;
                var zero = Vector256<double>.Zero;

                NativeBufferHelpers.Pack1dFp64Avx(leftMemoryPointer + tileStart, panel.A, count);
                NativeBufferHelpers.Pack1dFp64Avx(rightMemoryPointer + tileStart, panel.B, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
                {
                    var leftVector = Avx.LoadVector256(panel.A + i);
                    var rightVector = Avx.LoadVector256(panel.B + i);
                    var greaterThanOrEqualMask = Avx.CompareGreaterThanOrEqual(leftVector, rightVector);
                    var result = Avx.BlendVariable(zero, one, greaterThanOrEqualMask);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = panel.A[i] >= panel.B[i] ? 1f : 0f;
                }

                return panel;
            },
            data =>
            {
                NativeMemory.AlignedFree(data.A);
                NativeMemory.AlignedFree(data.B);
            });
    }

    private static unsafe void PointwiseLessThanAvxFp32(
        float* leftMemoryPointer,
        float* rightMemoryPointer,
        float* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp32 - 1) / NativeBufferHelpers.TileSizeFp32;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () =>
            {
                var a = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                var b = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                return new Panel2dFp32(a, b);
            },
            (tileIdx, _, panel) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp32;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp32, n);
                var count = tileEnd - tileStart;
                var one = Vector256<float>.One;
                var zero = Vector256<float>.Zero;

                NativeBufferHelpers.Pack1dFp32Avx(leftMemoryPointer + tileStart, panel.A, count);
                NativeBufferHelpers.Pack1dFp32Avx(rightMemoryPointer + tileStart, panel.B, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
                {
                    var leftVector = Avx.LoadVector256(panel.A + i);
                    var rightVector = Avx.LoadVector256(panel.B + i);
                    var lessThanMask = Avx.CompareLessThan(leftVector, rightVector);
                    var result = Avx.BlendVariable(zero, one, lessThanMask);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = panel.A[i] < panel.B[i] ? 1f : 0f;
                }

                return panel;
            },
            data =>
            {
                NativeMemory.AlignedFree(data.A);
                NativeMemory.AlignedFree(data.B);
            });
    }

    private static unsafe void PointwiseLessThanAvxFp64(
        double* leftMemoryPointer,
        double* rightMemoryPointer,
        double* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp64 - 1) / NativeBufferHelpers.TileSizeFp64;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () =>
            {
                var a = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                var b = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                return new Panel2dFp64(a, b);
            },
            (tileIdx, _, panel) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp64;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp64, n);
                var count = tileEnd - tileStart;
                var one = Vector256<double>.One;
                var zero = Vector256<double>.Zero;

                NativeBufferHelpers.Pack1dFp64Avx(leftMemoryPointer + tileStart, panel.A, count);
                NativeBufferHelpers.Pack1dFp64Avx(rightMemoryPointer + tileStart, panel.B, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
                {
                    var leftVector = Avx.LoadVector256(panel.A + i);
                    var rightVector = Avx.LoadVector256(panel.B + i);
                    var lessThanMask = Avx.CompareLessThan(leftVector, rightVector);
                    var result = Avx.BlendVariable(zero, one, lessThanMask);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = panel.A[i] < panel.B[i] ? 1f : 0f;
                }

                return panel;
            },
            data =>
            {
                NativeMemory.AlignedFree(data.A);
                NativeMemory.AlignedFree(data.B);
            });
    }

    private static unsafe void PointwiseLessThanOrEqualAvxFp32(
        float* leftMemoryPointer,
        float* rightMemoryPointer,
        float* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp32 - 1) / NativeBufferHelpers.TileSizeFp32;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () =>
            {
                var a = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                var b = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                return new Panel2dFp32(a, b);
            },
            (tileIdx, _, panel) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp32;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp32, n);
                var count = tileEnd - tileStart;
                var one = Vector256<float>.One;
                var zero = Vector256<float>.Zero;

                NativeBufferHelpers.Pack1dFp32Avx(leftMemoryPointer + tileStart, panel.A, count);
                NativeBufferHelpers.Pack1dFp32Avx(rightMemoryPointer + tileStart, panel.B, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
                {
                    var leftVector = Avx.LoadVector256(panel.A + i);
                    var rightVector = Avx.LoadVector256(panel.B + i);
                    var lessThanOrEqualMask = Avx.CompareLessThanOrEqual(leftVector, rightVector);
                    var result = Avx.BlendVariable(zero, one, lessThanOrEqualMask);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = panel.A[i] <= panel.B[i] ? 1f : 0f;
                }

                return panel;
            },
            data =>
            {
                NativeMemory.AlignedFree(data.A);
                NativeMemory.AlignedFree(data.B);
            });
    }

    private static unsafe void PointwiseLessThanOrEqualAvxFp64(
        double* leftMemoryPointer,
        double* rightMemoryPointer,
        double* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp64 - 1) / NativeBufferHelpers.TileSizeFp64;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () =>
            {
                var a = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                var b = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                return new Panel2dFp64(a, b);
            },
            (tileIdx, _, panel) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp64;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp64, n);
                var count = tileEnd - tileStart;
                var one = Vector256<double>.One;
                var zero = Vector256<double>.Zero;

                NativeBufferHelpers.Pack1dFp64Avx(leftMemoryPointer + tileStart, panel.A, count);
                NativeBufferHelpers.Pack1dFp64Avx(rightMemoryPointer + tileStart, panel.B, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
                {
                    var leftVector = Avx.LoadVector256(panel.A + i);
                    var rightVector = Avx.LoadVector256(panel.B + i);
                    var lessThanOrEqualMask = Avx.CompareLessThanOrEqual(leftVector, rightVector);
                    var result = Avx.BlendVariable(zero, one, lessThanOrEqualMask);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = panel.A[i] <= panel.B[i] ? 1f : 0f;
                }

                return panel;
            },
            data =>
            {
                NativeMemory.AlignedFree(data.A);
                NativeMemory.AlignedFree(data.B);
            });
    }
}