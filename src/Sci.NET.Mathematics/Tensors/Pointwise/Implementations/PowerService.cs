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

    public Tensor<TNumber> Pow<TNumber>(ITensor<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        return ServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Pow(value, power);
    }
}