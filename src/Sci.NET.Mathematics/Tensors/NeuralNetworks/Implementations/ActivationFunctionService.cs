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
            grad => grad.Multiply(value.SigmoidPrime()));

        return result;
    }

    public ITensor<TNumber> SigmoidPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);
        value.Backend.ActivationFunctions.SigmoidPrime(value, result);

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
            grad => grad.Multiply(value.ReLUPrime()));

        return result;
    }

    public ITensor<TNumber> ReLUPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);
        value.Backend.ActivationFunctions.ReLUPrime(value, result);

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
            grad => grad.Multiply(value.SoftmaxPrime()));

        return result;
    }

    public ITensor<TNumber> SoftmaxPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);
        using var softmaxValue = value.Softmax();
        value.Backend.ActivationFunctions.SoftmaxPrime(value, softmaxValue, result);

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
            grad => grad.Multiply(value.LeakyReLUPrime(alpha)));

        return result;
    }

    public ITensor<TNumber> LeakyReLUPrime<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.LeakyReLUPrime(value, result, alpha);

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
            grad => grad.Multiply(value.EluPrime(alpha)));

        return result;
    }

    public ITensor<TNumber> EluPrime<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.EluPrime(value, result, alpha);

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
            grad => grad.Multiply(value.CeluPrime(alpha)));

        return result;
    }

    public ITensor<TNumber> CeluPrime<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.CeluPrime(value, result, alpha);

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
            grad => grad.Multiply(value.SwishPrime()));

        return result;
    }

    public ITensor<TNumber> SwishPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.SwishPrime(value, result);

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
            grad => grad.Multiply(value.MishPrime()));

        return result;
    }

    public ITensor<TNumber> MishPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IHyperbolicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.MishPrime(value, result);

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
            grad => grad.Multiply(value.HardTanhPrime(min, max)));

        return result;
    }

    public ITensor<TNumber> HardTanhPrime<TNumber>(ITensor<TNumber> value, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.HardTanhPrime(
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
            grad => grad.Multiply(value.HardSigmoidPrime()));

        return result;
    }

    public ITensor<TNumber> HardSigmoidPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.HardSigmoidPrime(value, result);

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
            grad => grad.Multiply(value.LogSigmoidPrime()));

        return result;
    }

    public ITensor<TNumber> LogSigmoidPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.LogSigmoidPrime(value, result);

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
            grad => grad.Multiply(value.GELUPrime()));

        return result;
    }

    public ITensor<TNumber> GELUPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.GELUPrime(value, result);

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
            grad => grad.Multiply(value.SoftPlusPrime()));

        return result;
    }

    public ITensor<TNumber> SoftPlusPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.SoftPlusPrime(value, result);

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
            grad => grad.Multiply(value.SoftSignPrime()));

        return result;
    }

    public ITensor<TNumber> SoftSignPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend, false);

        value.Backend.ActivationFunctions.SoftSignPrime(value, result);

        return result;
    }
}