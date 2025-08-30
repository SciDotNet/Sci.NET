// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Common.Intrinsics;
using Sci.NET.Common.Performance;

namespace Sci.NET.Mathematics.Backends.Managed.BinaryOps;

internal class ManagedDivideOp : IManagedBinaryTensorOp
{
    public static bool IsAvxSupported<TNumber>()
    {
        return (IntrinsicsHelper.AvailableInstructionSets & SimdInstructionSet.Avx) != 0 &&
               (typeof(TNumber) == typeof(float) || typeof(TNumber) == typeof(double));
    }

    public static TNumber Invoke<TNumber>(TNumber left, TNumber right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return left / right;
    }

    [MethodImpl(ImplementationOptions.FastPath)]
    public static float Invoke(float left, float right)
    {
        return left / right;
    }

    [MethodImpl(ImplementationOptions.FastPath)]
    public static double Invoke(double left, double right)
    {
        return left / right;
    }

    [MethodImpl(ImplementationOptions.FastPath)]
    public static Vector256<float> InvokeAvx(Vector256<float> left, Vector256<float> right)
    {
        return Avx.Divide(left, right);
    }

    [MethodImpl(ImplementationOptions.FastPath)]
    public static Vector256<double> InvokeAvx(Vector256<double> left, Vector256<double> right)
    {
        return Avx.Divide(left, right);
    }
}