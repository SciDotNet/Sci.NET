// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Exceptions;
using Sci.NET.Mathematics.Tensors.Manipulation;

namespace Sci.NET.Mathematics.Tensors.Pointwise.Implementations;

[SuppressMessage("Performance", "CA1859:Use concrete types when possible for improved performance", Justification = "The concrete type is not known at compile time.")]
internal class ArithmeticService : IArithmeticService
{
    private readonly IDeviceGuardService _deviceGuardService;
    private readonly IGradientAppenderService _gradientAppenderService;
    private readonly IBroadcastService _broadcastService;

    public ArithmeticService()
    {
        _deviceGuardService = TensorServiceProvider.GetTensorOperationServiceProvider().GetDeviceGuardService();
        _gradientAppenderService = TensorServiceProvider.GetTensorOperationServiceProvider().GetGradientAppenderService();
        _broadcastService = TensorServiceProvider.GetTensorOperationServiceProvider().GetBroadcastingService();
    }

    public Scalar<TNumber> Add<TNumber>(Scalar<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAdd(left, right).ToScalar();
    }

    public Vector<TNumber> Add<TNumber>(Scalar<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAdd(left, right).ToVector();
    }

    public Matrix<TNumber> Add<TNumber>(Scalar<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAdd(left, right).ToMatrix();
    }

    public Tensor<TNumber> Add<TNumber>(Scalar<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAdd(left, right).ToTensor();
    }

    public Vector<TNumber> Add<TNumber>(Vector<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAdd(left, right).ToVector();
    }

    public Vector<TNumber> Add<TNumber>(Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAdd(left, right).ToVector();
    }

    public Matrix<TNumber> Add<TNumber>(Vector<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAdd(left, right).ToMatrix();
    }

    public Tensor<TNumber> Add<TNumber>(Vector<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAdd(left, right).ToTensor();
    }

    public Matrix<TNumber> Add<TNumber>(Matrix<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAdd(left, right).ToMatrix();
    }

    public Matrix<TNumber> Add<TNumber>(Matrix<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAdd(left, right).ToMatrix();
    }

    public Matrix<TNumber> Add<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAdd(left, right).ToMatrix();
    }

    public Tensor<TNumber> Add<TNumber>(Matrix<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAdd(left, right).ToTensor();
    }

    public Tensor<TNumber> Add<TNumber>(Tensor<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAdd(left, right).ToTensor();
    }

    public Tensor<TNumber> Add<TNumber>(Tensor<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAdd(left, right).ToTensor();
    }

    public Tensor<TNumber> Add<TNumber>(Tensor<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAdd(left, right).ToTensor();
    }

    public Tensor<TNumber> Add<TNumber>(Tensor<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAdd(left, right).ToTensor();
    }

    public ITensor<TNumber> Add<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAdd(left, right);
    }

    public Scalar<TNumber> Subtract<TNumber>(Scalar<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSubtract(left, right).ToScalar();
    }

    public Vector<TNumber> Subtract<TNumber>(Scalar<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSubtract(left, right).ToVector();
    }

    public Matrix<TNumber> Subtract<TNumber>(Scalar<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSubtract(left, right).ToMatrix();
    }

    public Tensor<TNumber> Subtract<TNumber>(Scalar<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSubtract(left, right).ToTensor();
    }

    public Vector<TNumber> Subtract<TNumber>(Vector<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSubtract(left, right).ToVector();
    }

    public Vector<TNumber> Subtract<TNumber>(Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSubtract(left, right).ToVector();
    }

    public Matrix<TNumber> Subtract<TNumber>(Vector<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSubtract(left, right).ToMatrix();
    }

    public Tensor<TNumber> Subtract<TNumber>(Vector<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSubtract(left, right).ToTensor();
    }

    public Matrix<TNumber> Subtract<TNumber>(Matrix<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSubtract(left, right).ToMatrix();
    }

    public Matrix<TNumber> Subtract<TNumber>(Matrix<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSubtract(left, right).ToMatrix();
    }

    public Matrix<TNumber> Subtract<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSubtract(left, right).ToMatrix();
    }

    public Tensor<TNumber> Subtract<TNumber>(Matrix<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSubtract(left, right).ToTensor();
    }

    public Tensor<TNumber> Subtract<TNumber>(Tensor<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSubtract(left, right).ToTensor();
    }

    public Tensor<TNumber> Subtract<TNumber>(Tensor<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSubtract(left, right).ToTensor();
    }

    public Tensor<TNumber> Subtract<TNumber>(Tensor<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSubtract(left, right).ToTensor();
    }

    public Tensor<TNumber> Subtract<TNumber>(Tensor<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSubtract(left, right).ToTensor();
    }

    public ITensor<TNumber> Subtract<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSubtract(left, right);
    }

    public Scalar<TNumber> Multiply<TNumber>(Scalar<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericMultiply(left, right).ToScalar();
    }

    public Vector<TNumber> Multiply<TNumber>(Scalar<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericMultiply(left, right).ToVector();
    }

    public Matrix<TNumber> Multiply<TNumber>(Scalar<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericMultiply(left, right).ToMatrix();
    }

    public Tensor<TNumber> Multiply<TNumber>(Scalar<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericMultiply(left, right).ToTensor();
    }

    public Vector<TNumber> Multiply<TNumber>(Vector<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericMultiply(left, right).ToVector();
    }

    public Vector<TNumber> Multiply<TNumber>(Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericMultiply(left, right).ToVector();
    }

    public Matrix<TNumber> Multiply<TNumber>(Vector<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericMultiply(left, right).ToMatrix();
    }

    public Tensor<TNumber> Multiply<TNumber>(Vector<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericMultiply(left, right).ToTensor();
    }

    public Matrix<TNumber> Multiply<TNumber>(Matrix<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericMultiply(left, right).ToMatrix();
    }

    public Matrix<TNumber> Multiply<TNumber>(Matrix<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericMultiply(left, right).ToMatrix();
    }

    public Matrix<TNumber> Multiply<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericMultiply(left, right).ToMatrix();
    }

    public Tensor<TNumber> Multiply<TNumber>(Matrix<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericMultiply(left, right).ToTensor();
    }

    public Tensor<TNumber> Multiply<TNumber>(Tensor<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericMultiply(left, right).ToTensor();
    }

    public Tensor<TNumber> Multiply<TNumber>(Tensor<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericMultiply(left, right).ToTensor();
    }

    public Tensor<TNumber> Multiply<TNumber>(Tensor<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericMultiply(left, right).ToTensor();
    }

    public Tensor<TNumber> Multiply<TNumber>(Tensor<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericMultiply(left, right).ToTensor();
    }

    public ITensor<TNumber> Multiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericMultiply(left, right);
    }

    public void MultiplyInplace<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        throw new NotSupportedException("Inplace multiplication is not supported.");
    }

    public Scalar<TNumber> Divide<TNumber>(Scalar<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericDivide(left, right).ToScalar();
    }

    public Vector<TNumber> Divide<TNumber>(Scalar<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericDivide(left, right).ToVector();
    }

    public Matrix<TNumber> Divide<TNumber>(Scalar<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericDivide(left, right).ToMatrix();
    }

    public Tensor<TNumber> Divide<TNumber>(Scalar<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericDivide(left, right).ToTensor();
    }

    public Vector<TNumber> Divide<TNumber>(Vector<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericDivide(left, right).ToVector();
    }

    public Vector<TNumber> Divide<TNumber>(Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericDivide(left, right).ToVector();
    }

    public Matrix<TNumber> Divide<TNumber>(Vector<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericDivide(left, right).ToMatrix();
    }

    public Tensor<TNumber> Divide<TNumber>(Vector<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericDivide(left, right).ToTensor();
    }

    public Matrix<TNumber> Divide<TNumber>(Matrix<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericDivide(left, right).ToMatrix();
    }

    public Matrix<TNumber> Divide<TNumber>(Matrix<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericDivide(left, right).ToMatrix();
    }

    public Matrix<TNumber> Divide<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericDivide(left, right).ToMatrix();
    }

    public Tensor<TNumber> Divide<TNumber>(Matrix<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericDivide(left, right).ToTensor();
    }

    public Tensor<TNumber> Divide<TNumber>(Tensor<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericDivide(left, right).ToTensor();
    }

    public Tensor<TNumber> Divide<TNumber>(Tensor<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericDivide(left, right).ToTensor();
    }

    public Tensor<TNumber> Divide<TNumber>(Tensor<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericDivide(left, right).ToTensor();
    }

    public Tensor<TNumber> Divide<TNumber>(Tensor<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericDivide(left, right).ToTensor();
    }

    public ITensor<TNumber> Divide<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericDivide(left, right);
    }

    public Scalar<TNumber> Negate<TNumber>(Scalar<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericNegate(value).ToScalar();
    }

    public Vector<TNumber> Negate<TNumber>(Vector<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericNegate(value).ToVector();
    }

    public Matrix<TNumber> Negate<TNumber>(Matrix<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericNegate(value).ToMatrix();
    }

    public Tensor<TNumber> Negate<TNumber>(Tensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericNegate(value).ToTensor();
    }

    public ITensor<TNumber> Negate<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericNegate(value);
    }

    public Scalar<TNumber> Abs<TNumber>(Scalar<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAbs(value).ToScalar();
    }

    public Vector<TNumber> Abs<TNumber>(Vector<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAbs(value).ToVector();
    }

    public Matrix<TNumber> Abs<TNumber>(Matrix<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAbs(value).ToMatrix();
    }

    public Tensor<TNumber> Abs<TNumber>(Tensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAbs(value).ToTensor();
    }

    public ITensor<TNumber> Abs<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericAbs(value);
    }

    public Scalar<TNumber> Sqrt<TNumber>(Scalar<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSqrt(value).ToScalar();
    }

    public Vector<TNumber> Sqrt<TNumber>(Vector<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSqrt(value).ToVector();
    }

    public Matrix<TNumber> Sqrt<TNumber>(Matrix<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSqrt(value).ToMatrix();
    }

    public Tensor<TNumber> Sqrt<TNumber>(Tensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSqrt(value).ToTensor();
    }

    public ITensor<TNumber> Sqrt<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return GenericSqrt(tensor);
    }

    private ITensor<TNumber> GenericAdd<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var device = _deviceGuardService.GuardBinaryOperation(left.Device, right.Device);
        var outputShape = GetBinaryOpOutputShape(left.Shape, right.Shape);
        var result = new Tensor<TNumber>(outputShape, device, left.RequiresGradient || right.RequiresGradient);

        device.Arithmetic.Add(left, right, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            left,
            right,
            null,
            grad => grad,
            grad => grad);

        return result;
    }

    private ITensor<TNumber> GenericSubtract<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var device = _deviceGuardService.GuardBinaryOperation(left.Device, right.Device);
        var outputShape = GetBinaryOpOutputShape(left.Shape, right.Shape);
        var result = new Tensor<TNumber>(outputShape, device, left.RequiresGradient || right.RequiresGradient);

        device.Arithmetic.Subtract(left, right, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            left,
            right,
            null,
            grad => grad,
            grad => grad.Negate());

        return result;
    }

    private ITensor<TNumber> GenericMultiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var device = _deviceGuardService.GuardBinaryOperation(left.Device, right.Device);
        var outputShape = GetBinaryOpOutputShape(left.Shape, right.Shape);
        var result = new Tensor<TNumber>(outputShape, device, left.RequiresGradient || right.RequiresGradient);

        device.Arithmetic.Multiply(left, right, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            left,
            right,
            null,
            right.Multiply,
            left.Multiply);

        return result;
    }

    private ITensor<TNumber> GenericDivide<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var device = _deviceGuardService.GuardBinaryOperation(left.Device, right.Device);
        var outputShape = GetBinaryOpOutputShape(left.Shape, right.Shape);
        var result = new Tensor<TNumber>(outputShape, device, left.RequiresGradient || right.RequiresGradient);

        device.Arithmetic.Divide(left, right, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            left,
            right,
            null,
            right.Divide,
            grad => grad.Multiply(left).Divide(right.Square()).Negate());

        return result;
    }

    private ITensor<TNumber> GenericNegate<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;

        if (!GenericMath.IsSigned<TNumber>())
        {
            var newMemoryBlock = value.Memory.Copy();

            var resultShortcut = new Tensor<TNumber>(newMemoryBlock, value.Shape, backend, value.RequiresGradient);

            _gradientAppenderService.AddGradientIfRequired(
                ref resultShortcut,
                value,
                null,
                grad => grad.Negate());

            return resultShortcut;
        }

        var result = new Tensor<TNumber>(backend, value.RequiresGradient, value.Shape.Dimensions);

        backend.Arithmetic.Negate(
            value.Memory,
            result.Memory,
            value.Shape.ElementCount);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad => grad.Negate());

        return result;
    }

    private ITensor<TNumber> GenericAbs<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;

        if (!GenericMath.IsSigned<TNumber>())
        {
            var newMemoryBlock = value.Memory.Copy();

            var resultShortcut = new Tensor<TNumber>(newMemoryBlock, value.Shape, backend, value.RequiresGradient);

            _gradientAppenderService.AddGradientIfRequired(
                ref resultShortcut,
                value,
                null,
                grad => grad);

            return resultShortcut;
        }

        var result = new Tensor<TNumber>(backend, value.RequiresGradient, value.Shape.Dimensions);

        backend.Arithmetic.Abs(
            value.Memory,
            result.Memory,
            value.Shape.ElementCount);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            value,
            null,
            grad =>
            {
                var gradResult = new Tensor<TNumber>(value.Shape, value.Backend);
                value.Backend.Arithmetic.AbsGradient(value.Memory, grad.Memory, gradResult.Memory, value.Shape.ElementCount);

                return gradResult;
            });

        return result;
    }

    private ITensor<TNumber> GenericSqrt<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = tensor.Backend;
        var result = new Tensor<TNumber>(backend, tensor.RequiresGradient, tensor.Shape.Dimensions);

        backend.Arithmetic.Sqrt(
            tensor.Memory,
            result.Memory,
            tensor.Shape.ElementCount);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad =>
            {
                using var two = new Scalar<TNumber>(TNumber.CreateChecked(2), backend);
                using var one = new Scalar<TNumber>(TNumber.CreateChecked(1), backend);
                using var localGradient = one.Divide(result.Multiply(two));

                return grad.Multiply(localGradient);
            });

        return result;
    }

    private Shape GetBinaryOpOutputShape(Shape left, Shape right)
    {
        Shape bigger, smaller;

        if (left.Rank < right.Rank)
        {
            bigger = right;
            smaller = left;
        }
        else
        {
            bigger = left;
            smaller = right;
        }

        if (!_broadcastService.CanBroadcastTo(smaller, bigger))
        {
            throw new InvalidShapeException($"Cannot broadcast shapes {left} and {right} for a binary operation.");
        }

        return bigger;
    }
}