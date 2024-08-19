// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Common;

namespace Sci.NET.Mathematics.Tensors.Pointwise.Implementations;

internal class PowerService : IPowerService
{
    private readonly IDeviceGuardService _guardService;

    public PowerService(ITensorOperationServiceProvider serviceProvider)
    {
        _guardService = serviceProvider.GetDeviceGuardService();
    }

    public Scalar<TNumber> Pow<TNumber>(Scalar<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(value.Device, power.Device);
        var backend = value.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Power.Pow(value, power, result);

        if (value.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                value,
                _ => TensorServiceProvider.GetTensorOperationServiceProvider().GetPowerService().PowDerivative(value, power));
        }

        return result;
    }

    public Vector<TNumber> Pow<TNumber>(Vector<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(value.Device, power.Device);
        var backend = value.Backend;
        var result = new Vector<TNumber>(value.Length, backend, requiresGradient: value.RequiresGradient);

        backend.Power.Pow(value, power, result);

        if (value.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                value,
                _ => TensorServiceProvider.GetTensorOperationServiceProvider().GetPowerService().PowDerivative(value, power));
        }

        return result;
    }

    public Matrix<TNumber> Pow<TNumber>(Matrix<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(value.Device, power.Device);
        var backend = value.Backend;
        var result = new Matrix<TNumber>(value.Rows, value.Columns, backend, requiresGradient: value.RequiresGradient);

        backend.Power.Pow(value, power, result);

        if (value.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                value,
                _ => TensorServiceProvider.GetTensorOperationServiceProvider().GetPowerService().PowDerivative(value, power));
        }

        return result;
    }

    public Tensor<TNumber> Pow<TNumber>(Tensor<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(value.Device, power.Device);
        var backend = value.Backend;
        var result = new Tensor<TNumber>(value.Shape, backend, requiresGradient: value.RequiresGradient);

        backend.Power.Pow(value, power, result);

        return result;
    }

    public ITensor<TNumber> PowDerivative<TNumber>(ITensor<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(value.Device, power.Device);
        var result = new Tensor<TNumber>(value.Shape, value.Backend, requiresGradient: false);
        var backend = value.Backend;
        backend.Power.PowDerivative(value, power, result);

        return result;
    }

    public Scalar<TNumber> Square<TNumber>(Scalar<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Power.Square(value, result);

        if (value.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                value,
                _ =>
                {
                    using var power = new Scalar<TNumber>(TNumber.CreateChecked(2), backend, requiresGradient: false);
                    return TensorServiceProvider.GetTensorOperationServiceProvider().GetArithmeticService().Multiply(value, power);
                });
        }

        return result;
    }

    public Vector<TNumber> Square<TNumber>(Vector<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Vector<TNumber>(value.Length, backend);

        backend.Power.Square(value, result);

        if (value.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                value,
                _ =>
                {
                    using var power = new Scalar<TNumber>(TNumber.CreateChecked(2), backend, requiresGradient: false);
                    return TensorServiceProvider.GetTensorOperationServiceProvider().GetArithmeticService().Multiply(value, power);
                });
        }

        return result;
    }

    public Matrix<TNumber> Square<TNumber>(Matrix<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Matrix<TNumber>(value.Rows, value.Columns, backend);

        backend.Power.Square(value, result);

        if (value.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                value,
                _ =>
                {
                    using var power = new Scalar<TNumber>(TNumber.CreateChecked(2), backend, requiresGradient: false);
                    return TensorServiceProvider.GetTensorOperationServiceProvider().GetArithmeticService().Multiply(value, power);
                });
        }

        return result;
    }

    public Tensor<TNumber> Square<TNumber>(Tensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Tensor<TNumber>(value.Shape, backend);

        backend.Power.Square(value, result);

        if (value.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                value,
                _ =>
                {
                    using var power = new Scalar<TNumber>(TNumber.CreateChecked(2), backend, requiresGradient: false);
                    return TensorServiceProvider.GetTensorOperationServiceProvider().GetArithmeticService().Multiply(value, power);
                });
        }

        return result;
    }

    public Scalar<TNumber> Exp<TNumber>(Scalar<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Power.Exp(value, result);

        if (value.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                value,
                _ => TensorServiceProvider.GetTensorOperationServiceProvider().GetPowerService().Exp(value));
        }

        return result;
    }

    public Vector<TNumber> Exp<TNumber>(Vector<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Vector<TNumber>(value.Length, backend);

        backend.Power.Exp(value, result);

        if (value.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                value,
                _ => TensorServiceProvider.GetTensorOperationServiceProvider().GetPowerService().Exp(value));
        }

        return result;
    }

    public Matrix<TNumber> Exp<TNumber>(Matrix<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Matrix<TNumber>(value.Rows, value.Columns, backend);

        backend.Power.Exp(value, result);

        if (value.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                value,
                _ => TensorServiceProvider.GetTensorOperationServiceProvider().GetPowerService().Exp(value));
        }

        return result;
    }

    public Tensor<TNumber> Exp<TNumber>(Tensor<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Tensor<TNumber>(value.Shape, backend);

        backend.Power.Exp(value, result);

        if (value.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                value,
                _ => TensorServiceProvider.GetTensorOperationServiceProvider().GetPowerService().Exp(value));
        }

        return result;
    }

    public ITensor<TNumber> Log<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = Tensor.CloneEmpty<ITensor<TNumber>, TNumber>(value);

        backend.Power.Log(value, result);

        if (value.RequiresGradient)
        {
            result.AddParent(
                value,
                _ => TensorServiceProvider.GetTensorOperationServiceProvider().GetPowerService().LogDerivative(value, TNumber.E));
        }

        return result;
    }

    public ITensor<TNumber> LogDerivative<TNumber>(ITensor<TNumber> value, TNumber logBase)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Tensor<TNumber>(value.Shape, backend, requiresGradient: false);

        backend.Power.LogDerivative(value, logBase, result);

        return result;
    }
}