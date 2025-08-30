// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Common.Intrinsics;
using Sci.NET.Mathematics.Backends.Managed.BinaryOps;

namespace Sci.NET.Mathematics.Backends.Iterators;

internal static class ManagedBinaryOpIterator
{
    public static unsafe void ApplyBlock1D<T, TOp>(
        T* leftPointer,
        T* rightPointer,
        T* resultPointer,
        long extent,
        long leftStride,
        long rightStride,
        long resultStride)
        where T : unmanaged, INumber<T>
        where TOp : IManagedBinaryTensorOp
    {
        if (CanAvx<T>() && resultStride == 1)
        {
            if (typeof(T) == typeof(float))
            {
                if (TOp.IsAvxSupported<float>())
                {
                    ApplyBlock1DAvxFp32<TOp>(
                        (float*)leftPointer,
                        (float*)rightPointer,
                        (float*)resultPointer,
                        extent,
                        leftStride,
                        rightStride);
                    return;
                }
            }
            else if (TOp.IsAvxSupported<double>())
            {
                ApplyBlock1DAvxFp64<TOp>(
                    (double*)leftPointer,
                    (double*)rightPointer,
                    (double*)resultPointer,
                    extent,
                    leftStride,
                    rightStride);
                return;
            }
        }

        for (long i = 0; i < extent; i++)
        {
            var a = Unsafe.Read<T>(leftPointer + (i * leftStride));
            var b = Unsafe.Read<T>(rightPointer + (i * rightStride));
            Unsafe.Write(resultPointer + (i * resultStride), TOp.Invoke(a, b));
        }
    }

    private static bool CanAvx<T>()
        where T : unmanaged, INumber<T>
    {
        return (IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
               Avx.IsSupported &&
               (typeof(T) == typeof(float) || typeof(T) == typeof(double));
    }

    private static unsafe void ApplyBlock1DAvxFp32<TOp>(
        float* leftPointer,
        float* rightPointer,
        float* resultPointer,
        long extent,
        long leftStride,
        long rightStride)
        where TOp : IManagedBinaryTensorOp
    {
        const int vectorCount = 8;
        long i = 0;

        if (leftStride == 1 && rightStride == 1)
        {
            for (; i <= extent - vectorCount; i += vectorCount)
            {
                var left = Avx.LoadVector256(leftPointer + i);
                var right = Avx.LoadVector256(rightPointer + i);
                var result = TOp.InvokeAvx(left, right);
                Avx.Store(resultPointer + i, result);
            }
        }
        else if (leftStride == 0 && rightStride == 1)
        {
            var leftScalar = Vector256.Create(*leftPointer);
            for (; i <= extent - vectorCount; i += vectorCount)
            {
                var right = Avx.LoadVector256(rightPointer + i);
                var result = TOp.InvokeAvx(leftScalar, right);
                Avx.Store(resultPointer + i, result);
            }
        }
        else if (leftStride == 1 && rightStride == 0)
        {
            var rightScalar = Vector256.Create(*rightPointer);
            for (; i <= extent - vectorCount; i += vectorCount)
            {
                var left = Avx.LoadVector256(leftPointer + i);
                var result = TOp.InvokeAvx(left, rightScalar);
                Avx.Store(resultPointer + i, result);
            }
        }

        for (; i < extent; i++)
        {
            var left = *(leftPointer + (i * leftStride));
            var right = *(rightPointer + (i * rightStride));
            *(resultPointer + i) = TOp.Invoke(left, right);
        }
    }

    private static unsafe void ApplyBlock1DAvxFp64<TOp>(
        double* leftPointer,
        double* rightPointer,
        double* resultPointer,
        long extent,
        long leftStride,
        long rightStride)
        where TOp : IManagedBinaryTensorOp
    {
        const int vectorCount = 4;
        long i = 0;

        if (leftStride == 1 && rightStride == 1)
        {
            for (; i <= extent - vectorCount; i += vectorCount)
            {
                var left = Avx.LoadVector256(leftPointer + i);
                var right = Avx.LoadVector256(rightPointer + i);
                var result = TOp.InvokeAvx(left, right);
                Avx.Store(resultPointer + i, result);
            }
        }
        else if (leftStride == 0 && rightStride == 1)
        {
            var leftScalar = Vector256.Create(*leftPointer);
            for (; i <= extent - vectorCount; i += vectorCount)
            {
                var right = Avx.LoadVector256(rightPointer + i);
                var result = TOp.InvokeAvx(leftScalar, right);
                Avx.Store(resultPointer + i, result);
            }
        }
        else if (leftStride == 1 && rightStride == 0)
        {
            var rightScalar = Vector256.Create(*rightPointer);
            for (; i <= extent - vectorCount; i += vectorCount)
            {
                var left = Avx.LoadVector256(leftPointer + i);
                var right = TOp.InvokeAvx(left, rightScalar);
                Avx.Store(resultPointer + i, right);
            }
        }

        for (; i < extent; i++)
        {
            var left = *(leftPointer + (i * leftStride));
            var right = *(rightPointer + (i * rightStride));
            *(resultPointer + i) = TOp.Invoke(left, right);
        }
    }
}