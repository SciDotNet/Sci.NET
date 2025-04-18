// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Common;

namespace Sci.NET.Mathematics.Tensors.Pointwise.Implementations;

internal class PowerService : IPowerService
{
    private readonly IDeviceGuardService _guardService;
    private readonly IGradientAppenderService _gradientAppenderService;

    public PowerService()
    {
        _guardService = TensorServiceProvider.GetTensorOperationServiceProvider().GetDeviceGuardService();
        _gradientAppenderService = TensorServiceProvider.GetTensorOperationServiceProvider().GetGradientAppenderService();
    }

    public ITensor<TNumber> Pow<TNumber>(ITensor<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        _ = _guardService.GuardBinaryOperation(value.Device, power.Device);
        var backend = value.Backend;
        var result = new Tensor<TNumber>(value.Shape, backend, value.RequiresGradient || power.RequiresGradient);

        backend.Power.Pow(value, power, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            power,
            null,
            grad => grad.Multiply(value.Pow(power.Subtract(TNumber.One))),
            grad => grad.Multiply(value.Pow(power).Multiply(value.Log())));

        return result;
    }

    public ITensor<TNumber> Square<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Tensor<TNumber>(value.Shape, backend, requiresGradient: value.RequiresGradient);

        backend.Power.Square(value, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad =>
            {
                using var two = new Scalar<TNumber>(TNumber.CreateChecked(2), backend, requiresGradient: false);
                return grad.Multiply(two).Multiply(value);
            });

        return result;
    }

    public ITensor<TNumber> Exp<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Tensor<TNumber>(value.Shape, backend, requiresGradient: value.RequiresGradient);

        backend.Power.Exp(value, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad => grad.Multiply(value.Exp()));

        return result;
    }

    public ITensor<TNumber> Log<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Tensor<TNumber>(value.Shape, backend, requiresGradient: value.RequiresGradient);

        backend.Power.Log(value, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad => grad.Multiply(LogDerivative(value)));

        return result;
    }

#pragma warning disable CA1822
    public ITensor<TNumber> LogDerivative<TNumber>(ITensor<TNumber> value)
#pragma warning restore CA1822
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Tensor<TNumber>(value.Shape, backend, requiresGradient: false);

        backend.Power.LogDerivative(value, result);

        return result;
    }
}