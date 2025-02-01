// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.Trigonometry.Implementations;

internal class TrigonometryService : ITrigonometryService
{
    private readonly IGradientAppenderService _gradientAppenderService;

    public TrigonometryService()
    {
        _gradientAppenderService = TensorServiceProvider.GetTensorOperationServiceProvider().GetGradientAppenderService();
    }

    public ITensor<TNumber> Sin<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);

        result.Backend.Trigonometry.Sin(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.SinBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Cos<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);
        result.Backend.Trigonometry.Cos(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.CosBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Tan<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);
        result.Backend.Trigonometry.Tan(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.TanBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Sin2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);
        result.Backend.Trigonometry.Sin2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Sin2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Cos2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);
        result.Backend.Trigonometry.Cos2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Cos2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Tan2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);
        result.Backend.Trigonometry.Tan2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Tan2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Sinh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);
        result.Backend.Trigonometry.Sinh(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.SinhBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Cosh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);
        result.Backend.Trigonometry.Cosh(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.CoshBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Tanh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend, tensor.RequiresGradient);
        result.Backend.Trigonometry.Tanh(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.TanhBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Sinh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Sinh2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Sinh2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Cosh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Cosh2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Cosh2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Tanh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Tanh2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Tanh2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Asin<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Asin(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.AsinBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Acos<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acos(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.AcosBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Atan<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Atan(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.AtanBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> ASinh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Asinh(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.AsinhBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> ACosh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acosh(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.AcoshBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> ATanh<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Atanh(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.AtanhBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Asin2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Asin2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Asin2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Acos2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acos2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Acos2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Atan2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Atan2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Atan2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> ASinh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Asinh2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Asinh2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> ACosh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acosh2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Acosh2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> ATanh2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Atanh2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Atanh2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Csc<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Csc(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.CscBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Sec<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Sec(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.SecBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Cot<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Cot(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.CotBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Csc2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Csc2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Csc2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Sec2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Sec2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Sec2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Cot2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Cot2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Cot2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Csch<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Csch(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.CschBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Sech<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Sech(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.SechBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Coth<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Coth(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.CothBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Csch2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Csch2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Csch2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Sech2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Sech2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Sech2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Coth2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Coth2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Coth2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Acsc<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acsc(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.AcscBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Asec<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Asec(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.AsecBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> Acot<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acot(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.AcotBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> ACsc2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acsc2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Acsc2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> ASec2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Asec2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Asec2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> ACot2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acot2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Acot2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> ACsch<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acsch(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.AcschBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> ASech<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Asech(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.AsechBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> ACoth<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acoth(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.AcothBackwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> ACsch2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acsch2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Acsch2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> ASech2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Asech2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Asech2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }

    public ITensor<TNumber> ACoth2<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IFloatingPointIeee754<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);
        result.Backend.Trigonometry.Acoth2(tensor, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                InvalidShapeException.ThrowIfDifferentElementCount(tensor.Shape, grad.Shape);
                var gradient = new Tensor<TNumber>(tensor.Shape, tensor.Backend, requiresGradient: false) { IsGradient = true };
                tensor.Backend.Trigonometry.Acoth2Backwards(tensor, grad, gradient);

                return gradient;
            });

        return result;
    }
}