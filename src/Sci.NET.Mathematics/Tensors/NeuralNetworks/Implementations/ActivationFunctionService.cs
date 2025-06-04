// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Common;

namespace Sci.NET.Mathematics.Tensors.NeuralNetworks.Implementations;

internal class ActivationFunctionService : IActivationFunctionService
{
    private readonly IGradientAppenderService _gradientAppenderService;

    public ActivationFunctionService()
    {
        _gradientAppenderService = TensorServiceProvider.GetTensorOperationServiceProvider().GetGradientAppenderService();
    }

    public ITensor<TNumber> Sigmoid<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, value.RequiresGradient);
        value.Backend.ActivationFunctions.Sigmoid(value, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad => grad.Multiply(value.SigmoidBackward()));

        return result;
    }

    public ITensor<TNumber> SigmoidBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);
        value.Backend.ActivationFunctions.SigmoidBackard(value, result);

        return result;
    }

    public ITensor<TNumber> ReLU<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, value.RequiresGradient);
        value.Backend.ActivationFunctions.ReLU(value, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad => grad.Multiply(value.ReLUBackward()));

        return result;
    }

    public ITensor<TNumber> ReLUBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);
        value.Backend.ActivationFunctions.ReLUBackward(value, result);

        return result;
    }

    public ITensor<TNumber> Softmax<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, value.RequiresGradient);
        using var sumBuffer = new Scalar<TNumber>(TNumber.Zero, backend: value.Backend, requiresGradient: false);

        value.Backend.ActivationFunctions.Softmax(value, sumBuffer, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad => grad.Multiply(value.SoftmaxBackward()));

        return result;
    }

    public ITensor<TNumber> SoftmaxBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);
        using var softmaxValue = value.Softmax();
        value.Backend.ActivationFunctions.SoftmaxBackward(value, softmaxValue, result);

        return result;
    }

    public ITensor<TNumber> LeakyReLU<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, value.RequiresGradient);

        value.Backend.ActivationFunctions.LeakyReLU(value, result, alpha);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad => grad.Multiply(value.LeakyReLUBackward(alpha)));

        return result;
    }

    public ITensor<TNumber> LeakyReLUBackward<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.LeakyReLUBackward(value, result, alpha);

        return result;
    }

    public ITensor<TNumber> Elu<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, value.RequiresGradient);

        value.Backend.ActivationFunctions.Elu(value, result, alpha);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad => grad.Multiply(value.EluBackward(alpha)));

        return result;
    }

    public ITensor<TNumber> EluBackward<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.EluBackward(value, result, alpha);

        return result;
    }

    public ITensor<TNumber> Celu<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, value.RequiresGradient);

        value.Backend.ActivationFunctions.Celu(value, result, alpha);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad => grad.Multiply(value.CeluBackward(alpha)));

        return result;
    }

    public ITensor<TNumber> CeluBackward<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.CeluBackward(value, result, alpha);

        return result;
    }

    public ITensor<TNumber> Swish<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, value.RequiresGradient);

        value.Backend.ActivationFunctions.Swish(value, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad => grad.Multiply(value.SwishBackward()));

        return result;
    }

    public ITensor<TNumber> SwishBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.SwishBackward(value, result);

        return result;
    }

    public ITensor<TNumber> Mish<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IHyperbolicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, value.RequiresGradient);

        value.Backend.ActivationFunctions.Mish(value, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad => grad.Multiply(value.MishBackward()));

        return result;
    }

    public ITensor<TNumber> MishBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IHyperbolicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.MishBackward(value, result);

        return result;
    }

    public ITensor<TNumber> HardTanh<TNumber>(ITensor<TNumber> value, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, value.RequiresGradient);

        value.Backend.ActivationFunctions.HardTanh(
            value,
            result,
            min,
            max);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad => grad.Multiply(value.HardTanhBackward(min, max)));

        return result;
    }

    public ITensor<TNumber> HardTanhBackward<TNumber>(ITensor<TNumber> value, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.HardTanhBackward(
            value,
            result,
            min,
            max);

        return result;
    }

    public ITensor<TNumber> HardSigmoid<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, value.RequiresGradient);

        value.Backend.ActivationFunctions.HardSigmoid(value, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad => grad.Multiply(value.HardSigmoidBackward()));

        return result;
    }

    public ITensor<TNumber> HardSigmoidBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.HardSigmoidBackward(value, result);

        return result;
    }

    public ITensor<TNumber> LogSigmoid<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, value.RequiresGradient);

        value.Backend.ActivationFunctions.LogSigmoid(value, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad => grad.Multiply(value.LogSigmoidBackward()));

        return result;
    }

    public ITensor<TNumber> LogSigmoidBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.LogSigmoidBackward(value, result);

        return result;
    }

    public ITensor<TNumber> GELU<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, value.RequiresGradient);

        value.Backend.ActivationFunctions.GELU(value, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad => grad.Multiply(value.GELUBackward()));

        return result;
    }

    public ITensor<TNumber> GELUBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.GELUBackward(value, result);

        return result;
    }

    public ITensor<TNumber> SoftPlus<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, value.RequiresGradient);

        value.Backend.ActivationFunctions.SoftPlus(value, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad => grad.Multiply(value.SoftPlusBackward()));

        return result;
    }

    public ITensor<TNumber> SoftPlusBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.SoftPlusBackward(value, result);

        return result;
    }

    public ITensor<TNumber> SoftSign<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, value.RequiresGradient);

        value.Backend.ActivationFunctions.SoftSign(value, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad => grad.Multiply(value.SoftSignBackward()));

        return result;
    }

    public ITensor<TNumber> SoftSignBackward<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.SoftSignBackward(value, result);

        return result;
    }
}