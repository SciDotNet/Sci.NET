// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Intrinsics;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Backends.Managed.Buffers;
using Sci.NET.Mathematics.Backends.Managed.Iterators;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedArithmeticKernels : IArithmeticKernels
{
    public void Add<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var iterator = new ManagedBinaryOpIterator<TNumber>(left, right, result);
        iterator.Apply((lOffset, rOffset, outOffset) => result.Memory[outOffset] = left.Memory[lOffset] + right.Memory[rOffset]);
    }

    public void Subtract<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var iterator = new ManagedBinaryOpIterator<TNumber>(left, right, result);
        iterator.Apply((lOffset, rOffset, outOffset) => result.Memory[outOffset] = left.Memory[lOffset] - right.Memory[rOffset]);
    }

    public void Multiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var iterator = new ManagedBinaryOpIterator<TNumber>(left, right, result);
        iterator.Apply((lOffset, rOffset, outOffset) => result.Memory[outOffset] = left.Memory[lOffset] * right.Memory[rOffset]);
    }

    public void Divide<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var iterator = new ManagedBinaryOpIterator<TNumber>(left, right, result);
        iterator.Apply((lOffset, rOffset, outOffset) => result.Memory[outOffset] = left.Memory[lOffset] / right.Memory[rOffset]);
    }

    public unsafe void Negate<TNumber>(
        IMemoryBlock<TNumber> tensor,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(float))
        {
            NegateAvxFp32(
                (float*)tensorBlock.Pointer,
                (float*)resultBlock.Pointer,
                n);
            return;
        }

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(double))
        {
            NegateAvxFp64(
                (double*)tensorBlock.Pointer,
                (double*)resultBlock.Pointer,
                n);
            return;
        }

        _ = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = -tensorBlock[i]);
    }

    public unsafe void Abs<TNumber>(
        IMemoryBlock<TNumber> tensor,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(float))
        {
            AbsAvxFp32(
                (float*)tensorBlock.Pointer,
                (float*)resultBlock.Pointer,
                n);
            return;
        }

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(double))
        {
            AbsAvxFp64(
                (double*)tensorBlock.Pointer,
                (double*)resultBlock.Pointer,
                n);
            return;
        }

        _ = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Abs(tensorBlock[i]));
    }

    public unsafe void AbsGradient<TNumber>(IMemoryBlock<TNumber> tensor, IMemoryBlock<TNumber> gradient, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor;
        var gradientBlock = (SystemMemoryBlock<TNumber>)gradient;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(float))
        {
            AbsBackwardAvxFp32(
                (float*)tensorBlock.Pointer,
                (float*)gradientBlock.Pointer,
                (float*)resultBlock.Pointer,
                n);
            return;
        }

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(double))
        {
            AbsBackwardAvxFp64(
                (double*)tensorBlock.Pointer,
                (double*)gradientBlock.Pointer,
                (double*)resultBlock.Pointer,
                n);
            return;
        }

        _ = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            1,
            i =>
            {
                var tensorValue = tensorBlock[i];
                var gradientValue = gradientBlock[i];

                if (tensorValue > TNumber.Zero)
                {
                    resultBlock[i] = gradientValue;
                }
#pragma warning disable IDE0045
                else if (tensorValue < TNumber.Zero)
#pragma warning restore IDE0045
                {
                    resultBlock[i] = TNumber.Zero - gradientValue;
                }
                else
                {
                    resultBlock[i] = TNumber.Zero;
                }
            });
    }

    public unsafe void AbsoluteDifference<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)left;
        var rightBlock = (SystemMemoryBlock<TNumber>)right;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(float))
        {
            AbsoluteDifferenceAvxFp32(
                (float*)leftBlock.Pointer,
                (float*)rightBlock.Pointer,
                (float*)resultBlock.Pointer,
                resultBlock.Length);
            return;
        }

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(double))
        {
            AbsoluteDifferenceAvxFp64(
                (double*)leftBlock.Pointer,
                (double*)rightBlock.Pointer,
                (double*)resultBlock.Pointer,
                resultBlock.Length);
            return;
        }

        _ = LazyParallelExecutor.For(
            0,
            resultBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Abs(leftBlock[i] - rightBlock[i]));
    }

    public unsafe void Sqrt<TNumber>(
        IMemoryBlock<TNumber> tensor,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(float))
        {
            SqrtAvxFp32(
                (float*)tensorBlock.Pointer,
                (float*)resultBlock.Pointer,
                n);
            return;
        }

        if ((IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
            typeof(TNumber) == typeof(double))
        {
            SqrtAvxFp64(
                (double*)tensorBlock.Pointer,
                (double*)resultBlock.Pointer,
                n);
            return;
        }

        _ = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = GenericMath.Sqrt(tensorBlock[i]));
    }

    private static unsafe void NegateAvxFp32(
        float* inputMemoryPointer,
        float* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp32 - 1) / NativeBufferHelpers.TileSizeFp32;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () => (nuint)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32),
            (tileIdx, _, data) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp32;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp32, n);
                var count = tileEnd - tileStart;
                var dataPtr = (float*)data.ToPointer();

                NativeBufferHelpers.Pack1dFp32Avx(inputMemoryPointer + tileStart, dataPtr, count);
                var minusOne = Vector256.Create(-1f);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
                {
                    var vector = Avx.LoadVector256(dataPtr + i);
                    var result = Avx.Multiply(vector, minusOne);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = -dataPtr[i];
                }

                return data;
            },
            data => NativeMemory.AlignedFree((float*)data));
    }

    private static unsafe void NegateAvxFp64(
        double* inputMemoryPointer,
        double* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp64 - 1) / NativeBufferHelpers.TileSizeFp64;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () => (nuint)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32),
            (tileIdx, _, data) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp64;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp64, n);
                var count = tileEnd - tileStart;
                var dataPtr = (double*)data.ToPointer();

                NativeBufferHelpers.Pack1dFp64Avx(inputMemoryPointer + tileStart, dataPtr, count);
                var minusOne = Vector256.Create(-1d);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
                {
                    var vector = Avx.LoadVector256(dataPtr + i);
                    var result = Avx.Multiply(vector, minusOne);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = -dataPtr[i];
                }

                return data;
            },
            data => NativeMemory.AlignedFree((double*)data));
    }

    private static unsafe void AbsAvxFp32(
        float* inputMemoryPointer,
        float* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp32 - 1) / NativeBufferHelpers.TileSizeFp32;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () => (nuint)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32),
            (tileIdx, _, data) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp32;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp32, n);
                var count = tileEnd - tileStart;
                var dataPtr = (float*)data.ToPointer();
                var mask256 = Vector256.Create(0x7FFF_FFFF).AsSingle();

                NativeBufferHelpers.Pack1dFp32Avx(inputMemoryPointer + tileStart, dataPtr, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
                {
                    var vector = Avx.LoadVector256(dataPtr + i);
                    var abs = Avx2.And(vector.AsInt32(), mask256.AsInt32()).AsSingle();

                    Avx.Store(outputMemoryPtr + tileStart + i, abs);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = Math.Abs(dataPtr[i]);
                }

                return data;
            },
            data => NativeMemory.AlignedFree((float*)data));
    }

    private static unsafe void AbsAvxFp64(
        double* inputMemoryPointer,
        double* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp64 - 1) / NativeBufferHelpers.TileSizeFp64;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () => (nuint)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32),
            (tileIdx, _, data) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp64;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp64, n);
                var count = tileEnd - tileStart;
                var dataPtr = (double*)data.ToPointer();
                var mask256 = Vector256.Create(0x7FFFFFFFFFFFFFFF).AsDouble();

                NativeBufferHelpers.Pack1dFp64Avx(inputMemoryPointer + tileStart, dataPtr, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
                {
                    var vector = Avx.LoadVector256(dataPtr + i);
                    var abs = Avx2.And(vector.AsInt64(), mask256.AsInt64()).AsDouble();

                    Avx.Store(outputMemoryPtr + tileStart + i, abs);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = Math.Abs(dataPtr[i]);
                }

                return data;
            },
            data => NativeMemory.AlignedFree((double*)data));
    }

    private static unsafe void AbsBackwardAvxFp32(
        float* inputMemoryPointer,
        float* gradientMemoryPtr,
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
                var panelA = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                var panelB = (float*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);

                return new Panel2dFp32(panelA, panelB);
            },
            (tileIdx, _, data) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp32;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp32, n);
                var count = tileEnd - tileStart;
                var minusOne = Vector256.Create(-1f);
                var zeroF = Vector256.Create(0f);

                NativeBufferHelpers.Pack1dFp32Avx(inputMemoryPointer + tileStart, data.A, count);
                NativeBufferHelpers.Pack1dFp32Avx(gradientMemoryPtr + tileStart, data.B, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
                {
                    var inputVector = Avx.LoadVector256(data.A + i);
                    var gradientVector = Avx.LoadVector256(data.B + i);
                    var positiveMask = Avx.CompareGreaterThan(inputVector, zeroF);
                    var negativeMask = Avx.CompareLessThan(inputVector, zeroF);
                    var positivePart = Avx.And(positiveMask, gradientVector);
                    var negativePart = Avx.Multiply(gradientVector, minusOne);
                    negativePart = Avx.And(negativeMask, negativePart);
                    var result = Avx.Or(positivePart, negativePart);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    var tensorValue = inputMemoryPointer[i];
                    var gradientValue = gradientMemoryPtr[i];

                    if (tensorValue > 0.0f)
                    {
                        outputMemoryPtr[i] = gradientValue;
                    }
#pragma warning disable IDE0045
                    else if (tensorValue < 0.0f)
#pragma warning restore IDE0045
                    {
                        outputMemoryPtr[i] = -gradientValue;
                    }
                    else
                    {
                        outputMemoryPtr[i] = 0.0f;
                    }
                }

                return data;
            },
            data =>
            {
                NativeMemory.AlignedFree(data.A);
                NativeMemory.AlignedFree(data.B);
            });
    }

    private static unsafe void AbsBackwardAvxFp64(
        double* inputMemoryPointer,
        double* gradientMemoryPtr,
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
                var panelA = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);
                var panelB = (double*)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32);

                return new Panel2dFp64(panelA, panelB);
            },
            (tileIdx, _, data) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp64;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp64, n);
                var count = tileEnd - tileStart;
                var minusOne = Vector256.Create(-1d);
                var zeroF = Vector256.Create(0d);

                NativeBufferHelpers.Pack1dFp64Avx(inputMemoryPointer + tileStart, data.A, count);
                NativeBufferHelpers.Pack1dFp64Avx(gradientMemoryPtr + tileStart, data.B, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
                {
                    var inputVector = Avx.LoadVector256(data.A + i);
                    var gradientVector = Avx.LoadVector256(data.B + i);
                    var positiveMask = Avx.CompareGreaterThan(inputVector, zeroF);
                    var negativeMask = Avx.CompareLessThan(inputVector, zeroF);
                    var positivePart = Avx.And(positiveMask, gradientVector);
                    var negativePart = Avx.Multiply(gradientVector, minusOne);
                    negativePart = Avx.And(negativeMask, negativePart);
                    var result = Avx.Or(positivePart, negativePart);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    var tensorValue = inputMemoryPointer[i];
                    var gradientValue = gradientMemoryPtr[i];

                    if (tensorValue > 0.0f)
                    {
                        outputMemoryPtr[i] = gradientValue;
                    }
#pragma warning disable IDE0045
                    else if (tensorValue < 0.0f)
#pragma warning restore IDE0045
                    {
                        outputMemoryPtr[i] = -gradientValue;
                    }
                    else
                    {
                        outputMemoryPtr[i] = 0.0f;
                    }
                }

                return data;
            },
            data =>
            {
                NativeMemory.AlignedFree(data.A);
                NativeMemory.AlignedFree(data.B);
            });
    }

    private static unsafe void AbsoluteDifferenceAvxFp32(
        float* leftMemoryPtr,
        float* rightMemoryPtr,
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
            (tileIdx, _, data) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp32;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp32, n);
                var count = tileEnd - tileStart;

                NativeBufferHelpers.Pack1dFp32Avx(leftMemoryPtr + tileStart, data.A, count);
                NativeBufferHelpers.Pack1dFp32Avx(rightMemoryPtr + tileStart, data.B, count);
                var mask256 = Vector256.Create(0x7FFF_FFFF).AsSingle();

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
                {
                    var leftVector = Avx.LoadVector256(data.A + i);
                    var rightVector = Avx.LoadVector256(data.B + i);
                    var difference = Avx.Subtract(leftVector, rightVector);
                    var abs = Avx2.And(difference.AsInt32(), mask256.AsInt32()).AsSingle();

                    Avx.Store(outputMemoryPtr + tileStart + i, abs);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = float.Abs(leftMemoryPtr[i] - rightMemoryPtr[i]);
                }

                return data;
            },
            data =>
            {
                NativeMemory.AlignedFree(data.A);
                NativeMemory.AlignedFree(data.B);
            });
    }

    private static unsafe void AbsoluteDifferenceAvxFp64(
        double* leftMemoryPtr,
        double* rightMemoryPtr,
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
            (tileIdx, _, data) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp64;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp64, n);
                var count = tileEnd - tileStart;

                NativeBufferHelpers.Pack1dFp64Avx(leftMemoryPtr + tileStart, data.A, count);
                NativeBufferHelpers.Pack1dFp64Avx(rightMemoryPtr + tileStart, data.B, count);
                var mask256 = Vector256.Create(0x7FFFFFFFFFFFFFFF).AsDouble();

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
                {
                    var leftVector = Avx.LoadVector256(data.A + i);
                    var rightVector = Avx.LoadVector256(data.B + i);
                    var difference = Avx.Subtract(leftVector, rightVector);
                    var abs = Avx2.And(difference.AsInt64(), mask256.AsInt64()).AsDouble();

                    Avx.Store(outputMemoryPtr + tileStart + i, abs);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = double.Abs(leftMemoryPtr[i] - rightMemoryPtr[i]);
                }

                return data;
            },
            data =>
            {
                NativeMemory.AlignedFree(data.A);
                NativeMemory.AlignedFree(data.B);
            });
    }

    private static unsafe void SqrtAvxFp32(
        float* inputMemoryPointer,
        float* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp32 - 1) / NativeBufferHelpers.TileSizeFp32;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () => (nuint)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32),
            (tileIdx, _, data) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp32;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp32, n);
                var count = tileEnd - tileStart;
                var dataPtr = (float*)data.ToPointer();

                NativeBufferHelpers.Pack1dFp32Avx(inputMemoryPointer + tileStart, dataPtr, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp32; i += NativeBufferHelpers.AvxVectorSizeFp32)
                {
                    var vector = Avx.LoadVector256(dataPtr + i);
                    var result = Avx.Sqrt(vector);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = float.Sqrt(dataPtr[i]);
                }

                return data;
            },
            data => NativeMemory.AlignedFree((float*)data));
    }

    private static unsafe void SqrtAvxFp64(
        double* inputMemoryPointer,
        double* outputMemoryPtr,
        long n)
    {
        var tileCount = (n + NativeBufferHelpers.TileSizeFp64 - 1) / NativeBufferHelpers.TileSizeFp64;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () => (nuint)NativeMemory.AlignedAlloc(NativeBufferHelpers.L1Size, 32),
            (tileIdx, _, data) =>
            {
                var tileStart = tileIdx * NativeBufferHelpers.TileSizeFp64;
                var tileEnd = Math.Min(tileStart + NativeBufferHelpers.TileSizeFp64, n);
                var count = tileEnd - tileStart;
                var dataPtr = (double*)data.ToPointer();

                NativeBufferHelpers.Pack1dFp64Avx(inputMemoryPointer + tileStart, dataPtr, count);

                var i = 0L;
                for (; i <= count - NativeBufferHelpers.AvxVectorSizeFp64; i += NativeBufferHelpers.AvxVectorSizeFp64)
                {
                    var vector = Avx.LoadVector256(dataPtr + i);
                    var result = Avx.Sqrt(vector);

                    Avx.Store(outputMemoryPtr + tileStart + i, result);
                }

                for (; i < count; ++i)
                {
                    outputMemoryPtr[tileStart + i] = double.Sqrt(dataPtr[i]);
                }

                return data;
            },
            data => NativeMemory.AlignedFree((double*)data));
    }
}