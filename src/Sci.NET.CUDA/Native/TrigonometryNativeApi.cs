// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.CUDA.Native.Extensions;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.CUDA.Native;

internal static class TrigonometryNativeApi
{
    public static unsafe void SinFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.SinFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void SinFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.SinFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void CosFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.CosFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void CosFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.CosFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void TanFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.TanFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void TanFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.TanFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Sin2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Sin2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Sin2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Sin2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Cos2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Cos2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Cos2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Cos2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Tan2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Tan2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Tan2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Tan2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void SinhFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.SinhFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void SinhFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.SinhFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void CoshFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.CoshFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void CoshFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.CoshFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void TanhFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.TanhFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void TanhFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.TanhFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Sinh2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Sinh2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Sinh2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Sinh2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Cosh2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Cosh2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Cosh2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Cosh2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Tanh2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Tanh2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Tanh2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Tanh2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void ASinFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AsinFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void ASinFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AsinFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void ACosFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AcosFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void ACosFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AcosFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void ATanFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AtanFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void ATanFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AtanFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Asin2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Asin2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Asin2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Asin2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Acos2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Acos2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Acos2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Acos2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Atan2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Atan2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Atan2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Atan2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AsinhFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AsinhFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AsinhFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AsinhFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AcoshFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AcoshFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AcoshFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AcoshFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AtanhFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AtanhFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AtanhFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AtanhFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Asinh2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Asinh2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Asinh2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Asinh2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Acosh2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Acosh2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Acosh2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Acosh2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Atanh2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Atanh2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Atanh2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Atanh2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void CscFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.CscFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void CscFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.CscFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void SecFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.SecFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void SecFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.SecFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void CotFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.CotFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void CotFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.CotFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Csc2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Csc2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Csc2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Csc2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Sec2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Sec2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Sec2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Sec2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Cot2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Cot2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Cot2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Cot2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void CschFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.CschFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void CschFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.CschFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void SechFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.SechFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void SechFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.SechFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void CothFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.CothFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void CothFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.CothFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Csch2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Csch2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Csch2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Csch2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Sech2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Sech2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Sech2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Sech2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Coth2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Coth2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Coth2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Coth2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AcscFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AcscFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AcscFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AcscFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AsecFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AsecFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AsecFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AsecFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AcotFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AcotFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AcotFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AcotFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Acsc2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Acsc2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Acsc2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Acsc2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Asec2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Asec2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Asec2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Asec2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Acot2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Acot2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Acot2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Acot2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AcscchFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AcscchFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AcscchFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AcscchFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AsechFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AsechFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AsechFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AsechFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AcothFp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AcothFp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void AcothFp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.AcothFp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Acscch2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Acscch2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Acscch2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Acscch2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Asech2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Asech2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Asech2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Asech2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Acoth2Fp32<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Acoth2Fp32(
                (float*)tensor.Handle.ToPointer(),
                (float*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }

    public static unsafe void Acoth2Fp64<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        TrigonometryNativeMethods.Acoth2Fp64(
                (double*)tensor.Handle.ToPointer(),
                (double*)result.Handle.ToPointer(),
                tensor.Shape.ElementCount)
            .Guard();
    }
}