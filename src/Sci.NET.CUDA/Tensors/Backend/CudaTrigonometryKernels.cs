// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.CUDA.Native;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.CUDA.Tensors.Backend;

internal class CudaTrigonometryKernels : ITrigonometryKernels
{
    public void Sin<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.SinFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.SinFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Cos<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.CosFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.CosFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Tan<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.TanFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.TanFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Sin2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Sin2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Sin2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Cos2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Cos2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Cos2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Tan2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Tan2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Tan2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Sinh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.SinhFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.SinhFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Cosh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.CoshFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.CoshFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Tanh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.TanhFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.TanhFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Sinh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Sinh2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Sinh2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Cosh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Cosh2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Cosh2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Tanh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Tanh2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Tanh2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Asin<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.ASinFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.ASinFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Acos<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.ACosFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.ACosFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Atan<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.ATanFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.ATanFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Asin2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Asin2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Asin2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Acos2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Acos2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Acos2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Atan2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Atan2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Atan2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Asinh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.AsinhFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.AsinhFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Acosh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.AcoshFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.AcoshFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Atanh<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.AtanhFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.AtanhFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Asinh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Asinh2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Asinh2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Acosh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Acosh2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Acosh2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Atanh2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Atanh2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Atanh2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Csc<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.CscFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.CscFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Sec<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.SecFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.SecFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Cot<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.CotFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.CotFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Csc2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Csc2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Csc2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Sec2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Sec2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Sec2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Cot2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Cot2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Cot2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Csch<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.CschFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.CschFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Sech<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.SechFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.SechFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Coth<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.CothFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.CothFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Csch2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Csch2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Csch2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Sech2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Sech2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Sech2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Coth2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Coth2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Coth2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Acsc<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.AcscFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.AcscFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Asec<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.AsecFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.AsecFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Acot<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.AcotFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.AcotFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Acsc2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Acsc2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Acsc2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Asec2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Asec2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Asec2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Acot2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Acot2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Acot2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for trigonometric functions.");
        }
    }

    public void Acsch<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.AcscchFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.AcscchFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Asech<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.AsechFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.AsechFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Acoth<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.AcothFp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.AcothFp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Acsch2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Acscch2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Acscch2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Asech2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Asech2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Asech2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }

    public void Acoth2<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        switch (TNumber.Zero)
        {
            case float:
                TrigonometryNativeApi.Acoth2Fp32(tensor, result);
                break;
            case double:
                TrigonometryNativeApi.Acoth2Fp64(tensor, result);
                break;
            default:
                throw new NotSupportedException("Unsupported type for hyperbolic functions.");
        }
    }
}