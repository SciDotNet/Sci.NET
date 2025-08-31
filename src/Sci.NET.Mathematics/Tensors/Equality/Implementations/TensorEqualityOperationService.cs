// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.Equality.Implementations;

internal class TensorEqualityOperationService : ITensorEqualityOperationService
{
    private readonly IDeviceGuardService _guardService;
    private readonly IGradientAppenderService _gradientAppenderService;

    public TensorEqualityOperationService()
    {
        _guardService = TensorServiceProvider.GetTensorOperationServiceProvider().GetDeviceGuardService();
        _gradientAppenderService = TensorServiceProvider.GetTensorOperationServiceProvider().GetGradientAppenderService();
    }

    public ITensor<TNumber> PointwiseEquals<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        InvalidShapeException.ThrowIfDifferentElementCount(left.Shape, right.Shape);
        var backend = _guardService.GuardBinaryOperation(left.Device, right.Device);

        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.EqualityOperations.PointwiseEqualsKernel(left.Memory, right.Memory, result.Memory, left.Shape.ElementCount);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            left,
            right,
            null,
            grad => grad,
            grad => grad);

        return result;
    }

    public ITensor<TNumber> PointwiseNotEqual<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        InvalidShapeException.ThrowIfDifferentElementCount(left.Shape, right.Shape);
        var backend = _guardService.GuardBinaryOperation(left.Device, right.Device);

        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.EqualityOperations.PointwiseNotEqualKernel(left.Memory, right.Memory, result.Memory, left.Shape.ElementCount);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            left,
            right,
            null,
            grad => grad,
            grad => grad);

        return result;
    }

    public ITensor<TNumber> PointwiseGreaterThan<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        InvalidShapeException.ThrowIfDifferentElementCount(left.Shape, right.Shape);
        var backend = _guardService.GuardBinaryOperation(left.Device, right.Device);

        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.EqualityOperations.PointwiseGreaterThanKernel(left.Memory, right.Memory, result.Memory, left.Shape.ElementCount);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            left,
            right,
            null,
            grad => grad,
            grad => grad);

        return result;
    }

    public ITensor<TNumber> PointwiseGreaterThanOrEqual<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        InvalidShapeException.ThrowIfDifferentElementCount(left.Shape, right.Shape);
        var backend = _guardService.GuardBinaryOperation(left.Device, right.Device);

        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.EqualityOperations.PointwiseGreaterThanOrEqualKernel(left.Memory, right.Memory, result.Memory, left.Shape.ElementCount);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            left,
            right,
            null,
            grad => grad,
            grad => grad);

        return result;
    }

    public ITensor<TNumber> PointwiseLessThan<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        InvalidShapeException.ThrowIfDifferentElementCount(left.Shape, right.Shape);
        var backend = _guardService.GuardBinaryOperation(left.Device, right.Device);

        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.EqualityOperations.PointwiseLessThanKernel(left.Memory, right.Memory, result.Memory, left.Shape.ElementCount);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            left,
            right,
            null,
            grad => grad,
            grad => grad);

        return result;
    }

    public ITensor<TNumber> PointwiseLessThanOrEqual<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        InvalidShapeException.ThrowIfDifferentElementCount(left.Shape, right.Shape);
        var backend = _guardService.GuardBinaryOperation(left.Device, right.Device);

        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.EqualityOperations.PointwiseLessThanOrEqualKernel(left.Memory, right.Memory, result.Memory, left.Shape.ElementCount);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            left,
            right,
            null,
            grad => grad,
            grad => grad);

        return result;
    }
}