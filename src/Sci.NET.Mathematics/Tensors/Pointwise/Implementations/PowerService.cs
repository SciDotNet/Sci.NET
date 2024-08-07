﻿// Copyright (c) Sci.NET Foundation. All rights reserved.
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

        return result;
    }

    public Vector<TNumber> Pow<TNumber>(Vector<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(value.Device, power.Device);
        var backend = value.Backend;
        var result = new Vector<TNumber>(value.Length, backend);

        backend.Power.Pow(value, power, result);

        return result;
    }

    public Matrix<TNumber> Pow<TNumber>(Matrix<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(value.Device, power.Device);
        var backend = value.Backend;
        var result = new Matrix<TNumber>(value.Rows, value.Columns, backend);

        backend.Power.Pow(value, power, result);

        return result;
    }

    public Tensor<TNumber> Pow<TNumber>(Tensor<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(value.Device, power.Device);
        var backend = value.Backend;
        var result = new Tensor<TNumber>(value.Shape, backend);

        backend.Power.Pow(value, power, result);

        return result;
    }

    public Scalar<TNumber> Square<TNumber>(Scalar<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Power.Square(value, result);

        return result;
    }

    public Vector<TNumber> Square<TNumber>(Vector<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Vector<TNumber>(value.Length, backend);

        backend.Power.Square(value, result);

        return result;
    }

    public Matrix<TNumber> Square<TNumber>(Matrix<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Matrix<TNumber>(value.Rows, value.Columns, backend);

        backend.Power.Square(value, result);

        return result;
    }

    public Tensor<TNumber> Square<TNumber>(Tensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Tensor<TNumber>(value.Shape, backend);

        backend.Power.Square(value, result);

        return result;
    }

    public Scalar<TNumber> Exp<TNumber>(Scalar<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Power.Exp(value, result);

        return result;
    }

    public Vector<TNumber> Exp<TNumber>(Vector<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Vector<TNumber>(value.Length, backend);

        backend.Power.Exp(value, result);

        return result;
    }

    public Matrix<TNumber> Exp<TNumber>(Matrix<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Matrix<TNumber>(value.Rows, value.Columns, backend);

        backend.Power.Exp(value, result);

        return result;
    }

    public Tensor<TNumber> Exp<TNumber>(Tensor<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Tensor<TNumber>(value.Shape, backend);

        backend.Power.Exp(value, result);

        return result;
    }

    public ITensor<TNumber> Log<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = Tensor.CloneEmpty<ITensor<TNumber>, TNumber>(value);

        backend.Power.Log(value, result);

        return result;
    }
}