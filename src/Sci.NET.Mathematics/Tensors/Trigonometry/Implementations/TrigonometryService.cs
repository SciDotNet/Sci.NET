// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Trigonometry.Implementations;

internal class TrigonometryService : ITrigonometryService
{
    public ITensor<TNumber> Sin<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);

        result.Backend.Trigonometry.Sin(tensor, result);

        if (tensor.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(tensor, _ => tensor.Cos());
        }

        return result;
    }

    public ITensor<TNumber> Cos<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);
        result.Backend.Trigonometry.Cos(tensor, result);

        if (tensor.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(tensor, _ => tensor.Sin().Negate());
        }

        return result;
    }

    public ITensor<TNumber> Tan<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);
        result.Backend.Trigonometry.Tan(tensor, result);

        if (tensor.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(tensor, _ => tensor.Sec2());
        }

        return result;
    }

    public ITensor<TNumber> Sin2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);
        result.Backend.Trigonometry.Sin2(tensor, result);

        if (tensor.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                tensor,
                _ => TensorServiceProvider.GetTensorOperationServiceProvider().GetTrigonometryService().Sin2Backwards(tensor));
        }

        return result;
    }

    public ITensor<TNumber> Sin2Backwards<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, false);
        result.Backend.Trigonometry.Sin2Backwards(tensor, result, tensor.Shape.ElementCount);

        return result;
    }

    public ITensor<TNumber> Cos2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);
        result.Backend.Trigonometry.Cos2(tensor, result);

        if (tensor.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                tensor,
                _ => TensorServiceProvider.GetTensorOperationServiceProvider().GetTrigonometryService().Cos2Backwards(tensor));
        }

        return result;
    }

    public ITensor<TNumber> Cos2Backwards<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, false);
        result.Backend.Trigonometry.Cos2Backwards(tensor, result, tensor.Shape.ElementCount);

        return result;
    }

    public ITensor<TNumber> Tan2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);
        result.Backend.Trigonometry.Tan2(tensor, result);

        if (tensor.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                tensor,
                _ => TensorServiceProvider.GetTensorOperationServiceProvider().GetTrigonometryService().Tan2Backwards(tensor));
        }

        return result;
    }

    public ITensor<TNumber> Tan2Backwards<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, false);
        result.Backend.Trigonometry.Tan2Backwards(tensor, result, tensor.Shape.ElementCount);

        return result;
    }

    public ITensor<TNumber> Sinh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);
        result.Backend.Trigonometry.Sinh(tensor, result);

        if (tensor.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(tensor, _ => tensor.Cosh());
        }

        return result;
    }

    public ITensor<TNumber> Cosh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);
        result.Backend.Trigonometry.Cosh(tensor, result);

        if (tensor.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(tensor, _ => tensor.Sinh());
        }

        return result;
    }

    public ITensor<TNumber> Tanh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);
        result.Backend.Trigonometry.Tanh(tensor, result);

        if (tensor.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                tensor,
                _ => TensorServiceProvider.GetTensorOperationServiceProvider().GetTrigonometryService().TanhBackwards(tensor));
        }

        return result;
    }

    public ITensor<TNumber> TanhBackwards<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, false);
        result.Backend.Trigonometry.TanhBackwards(tensor, result, tensor.Shape.ElementCount);

        return result;
    }

    public ITensor<TNumber> Sinh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Sinh2(tensor, result);
        return result;
    }

    public ITensor<TNumber> Cosh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Cosh2(tensor, result);
        return result;
    }

    public ITensor<TNumber> Tanh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Tanh2(tensor, result);
        return result;
    }

    public ITensor<TNumber> Asin<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Asin(tensor, result);
        return result;
    }

    public ITensor<TNumber> Acos<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acos(tensor, result);
        return result;
    }

    public ITensor<TNumber> Atan<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Atan(tensor, result);
        return result;
    }

    public ITensor<TNumber> ASinh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Asinh(tensor, result);
        return result;
    }

    public ITensor<TNumber> ACosh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acosh(tensor, result);
        return result;
    }

    public ITensor<TNumber> ATanh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Atanh(tensor, result);
        return result;
    }

    public ITensor<TNumber> Asin2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Asin2(tensor, result);
        return result;
    }

    public ITensor<TNumber> Acos2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acos2(tensor, result);
        return result;
    }

    public ITensor<TNumber> Atan2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Atan2(tensor, result);
        return result;
    }

    public ITensor<TNumber> ASinh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Asinh2(tensor, result);
        return result;
    }

    public ITensor<TNumber> ACosh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acosh2(tensor, result);
        return result;
    }

    public ITensor<TNumber> ATanh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Atanh2(tensor, result);
        return result;
    }

    public ITensor<TNumber> Csc<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Csc(tensor, result);
        return result;
    }

    public ITensor<TNumber> Sec<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Sec(tensor, result);
        return result;
    }

    public ITensor<TNumber> Cot<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Cot(tensor, result);
        return result;
    }

    public ITensor<TNumber> Csc2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Csc2(tensor, result);
        return result;
    }

    public ITensor<TNumber> Sec2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Sec2(tensor, result);
        return result;
    }

    public ITensor<TNumber> Cot2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Cot2(tensor, result);
        return result;
    }

    public ITensor<TNumber> Csch<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Csch(tensor, result);
        return result;
    }

    public ITensor<TNumber> Sech<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Sech(tensor, result);
        return result;
    }

    public ITensor<TNumber> Coth<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Coth(tensor, result);
        return result;
    }

    public ITensor<TNumber> Csch2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Csch2(tensor, result);
        return result;
    }

    public ITensor<TNumber> Sech2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Sech2(tensor, result);
        return result;
    }

    public ITensor<TNumber> Coth2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Coth2(tensor, result);
        return result;
    }

    public ITensor<TNumber> Acsc<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acsc(tensor, result);
        return result;
    }

    public ITensor<TNumber> Asec<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Asec(tensor, result);
        return result;
    }

    public ITensor<TNumber> Acot<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acot(tensor, result);
        return result;
    }

    public ITensor<TNumber> ACsc2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acsc2(tensor, result);
        return result;
    }

    public ITensor<TNumber> ASec2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Asec2(tensor, result);
        return result;
    }

    public ITensor<TNumber> ACot2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acot2(tensor, result);
        return result;
    }

    public ITensor<TNumber> ACsch<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acsch(tensor, result);
        return result;
    }

    public ITensor<TNumber> ASech<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Asech(tensor, result);
        return result;
    }

    public ITensor<TNumber> ACoth<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acoth(tensor, result);
        return result;
    }

    public ITensor<TNumber> ACsch2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acsch2(tensor, result);
        return result;
    }

    public ITensor<TNumber> ASech2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Asech2(tensor, result);
        return result;
    }

    public ITensor<TNumber> ACoth2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acoth2(tensor, result);
        return result;
    }
}