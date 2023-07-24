// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Exceptions;
using Sci.NET.Mathematics.Tensors.Manipulation;

namespace Sci.NET.Mathematics.Tensors.Pointwise.Implementations;

internal class ArithmeticService : IArithmeticService
{
    private readonly IDeviceGuardService _guardService;
    private readonly IBroadcastService _broadcastService;

    public ArithmeticService(ITensorOperationServiceProvider provider)
    {
        _guardService = provider.GetDeviceGuardService();
        _broadcastService = provider.GetBroadcastingService();
    }

    public Scalar<TNumber> Add<TNumber>(Scalar<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Arithmetic.Add(left, right, result);

        return result;
    }

    public Vector<TNumber> Add<TNumber>(Scalar<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Vector<TNumber>(right.Length, backend);

        backend.Arithmetic.Add(left, right, result);

        return result;
    }

    public Matrix<TNumber> Add<TNumber>(Scalar<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.Add(left, right, result);

        return result;
    }

    public Tensor<TNumber> Add<TNumber>(Scalar<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Tensor<TNumber>(right.Shape, backend);

        backend.Arithmetic.Add(left, right, result);

        return result;
    }

    public Vector<TNumber> Add<TNumber>(Vector<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Vector<TNumber>(left.Length, backend);

        backend.Arithmetic.Add(left, right, result);

        return result;
    }

    public Vector<TNumber> Add<TNumber>(Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Length != right.Length)
        {
            if (!_broadcastService.CanBroadcastBinaryOp(left.Shape, right.Shape))
            {
                throw new InvalidShapeException(
                    $"Cannot add shapes {left.Shape} and {right.Shape}.");
            }

            var (newLeft, newRight) = _broadcastService.Broadcast(left, right);

            return (newLeft + newRight).AsVector();
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Vector<TNumber>(left.Length, backend);

        backend.Arithmetic.Add(left, right, result);

        return result;
    }

    public Matrix<TNumber> Add<TNumber>(Vector<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Length != right.Rows)
        {
            if (!_broadcastService.CanBroadcastBinaryOp(left.Shape, right.Shape))
            {
                throw new InvalidShapeException(
                    $"Cannot add shapes {left.Shape} and {right.Shape}.");
            }

            var (newLeft, newRight) = _broadcastService.Broadcast(left, right);

            return (newLeft + newRight).AsMatrix();
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.Add(left, right, result);

        return result;
    }

    public Matrix<TNumber> Add<TNumber>(Matrix<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);

        var backend = left.Backend;
        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.Add(left, right, result);

        return result;
    }

    public Matrix<TNumber> Add<TNumber>(Matrix<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Rows != right.Length)
        {
            if (!_broadcastService.CanBroadcastBinaryOp(left.Shape, right.Shape))
            {
                throw new InvalidShapeException(
                    $"Cannot add shapes {left.Shape} and {right.Shape}.");
            }

            var (newLeft, newRight) = _broadcastService.Broadcast(left, right);

            return (newLeft + newRight).AsMatrix();
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.Add(left, right, result);

        return result;
    }

    public Matrix<TNumber> Add<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Rows != right.Rows || left.Columns != right.Columns)
        {
            if (!_broadcastService.CanBroadcastBinaryOp(left.Shape, right.Shape))
            {
                throw new InvalidShapeException(
                    $"Cannot add shapes {left.Shape} and {right.Shape}.");
            }

            var (newLeft, newRight) = _broadcastService.Broadcast(left, right);

            return (newLeft + newRight).AsMatrix();
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.Add(left, right, result);

        return result;
    }

    public Tensor<TNumber> Add<TNumber>(Tensor<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.Add(left, right, result);

        return result;
    }

    public Tensor<TNumber> Add<TNumber>(Tensor<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (!left.Shape.Equals(right.Shape))
        {
            if (!_broadcastService.CanBroadcastBinaryOp(left.Shape, right.Shape))
            {
                throw new InvalidShapeException(
                    $"Cannot add shapes {left.Shape} and {right.Shape}.");
            }

            var (newLeft, newRight) = _broadcastService.Broadcast(left, right);

            return (newLeft + newRight).AsVector();
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.Add(left, right, result);

        return result;
    }

    // ReSharper disable once CyclomaticComplexity
    public ITensor<TNumber> Add<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.IsScalar())
        {
            var leftScalar = left.AsScalar();

            if (right.IsScalar())
            {
                return leftScalar.Add(right.AsScalar());
            }

            if (right.IsVector())
            {
                return leftScalar.Add(right.AsVector());
            }

            if (right.IsMatrix())
            {
                return leftScalar.Add(right.AsMatrix());
            }

            return leftScalar.Add(right.AsTensor());
        }

        if (left.IsVector())
        {
            var leftVector = left.AsVector();

            if (right.IsScalar())
            {
                return leftVector.Add(right.AsScalar());
            }

            if (right.IsVector())
            {
                return leftVector.Add(right.AsVector());
            }

            if (right.IsMatrix())
            {
                return leftVector.Add(right.AsMatrix());
            }

            return leftVector.Add(right.AsTensor());
        }

        if (left.IsMatrix())
        {
            var leftMatrix = left.AsMatrix();

            if (right.IsScalar())
            {
                return leftMatrix.Add(right.AsScalar());
            }

            if (right.IsVector())
            {
                return leftMatrix.Add(right.AsVector());
            }

            if (right.IsMatrix())
            {
                return leftMatrix.Add(right.AsMatrix());
            }

            return leftMatrix.Add(right.AsTensor());
        }

        if (left.IsTensor())
        {
            var leftTensor = left.AsTensor();

            if (right.IsScalar())
            {
                return leftTensor.Add(right.AsScalar());
            }

            if (right.IsVector())
            {
                return leftTensor.Add(right.AsVector());
            }

            if (right.IsMatrix())
            {
                return leftTensor.Add(right.AsMatrix());
            }

            return leftTensor.Add(right.AsTensor());
        }

        throw new UnreachableException();
    }

    public Scalar<TNumber> Subtract<TNumber>(Scalar<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Arithmetic.Subtract(left, right, result);

        return result;
    }

    public Vector<TNumber> Subtract<TNumber>(Scalar<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Vector<TNumber>(right.Length, backend);

        backend.Arithmetic.Subtract(left, right, result);

        return result;
    }

    public Matrix<TNumber> Subtract<TNumber>(Scalar<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.Subtract(left, right, result);

        return result;
    }

    public Tensor<TNumber> Subtract<TNumber>(Scalar<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Tensor<TNumber>(right.Shape, backend);

        backend.Arithmetic.Subtract(left, right, result);

        return result;
    }

    public Vector<TNumber> Subtract<TNumber>(Vector<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Vector<TNumber>(left.Length, backend);

        backend.Arithmetic.Subtract(left, right, result);

        return result;
    }

    public Vector<TNumber> Subtract<TNumber>(Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Length != right.Length)
        {
            throw new InvalidShapeException(
                $"The length of the left vector {left.Shape} must match the length of the right vector {right.Shape}.");
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Vector<TNumber>(left.Length, backend);

        backend.Arithmetic.Subtract(left, right, result);

        return result;
    }

    public Matrix<TNumber> Subtract<TNumber>(Vector<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Length != right.Rows)
        {
            throw new InvalidShapeException(
                $"The length of the left vector {left.Shape} must match the length of the right vector {right.Shape}.");
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.Subtract(left, right, result);

        return result;
    }

    public Matrix<TNumber> Subtract<TNumber>(Matrix<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);

        var backend = left.Backend;
        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.Subtract(left, right, result);

        return result;
    }

    public Matrix<TNumber> Subtract<TNumber>(Matrix<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Rows != right.Length)
        {
            throw new InvalidShapeException(
                $"The length of the left vector {left.Shape} must match the length of the right vector {right.Shape}.");
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.Subtract(left, right, result);

        return result;
    }

    public Matrix<TNumber> Subtract<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Rows != right.Rows || left.Columns != right.Columns)
        {
            throw new InvalidShapeException(
                $"The shape of the left matrix {left.Shape} must match the shape of the right matrix {right.Shape}.");
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.Subtract(left, right, result);

        return result;
    }

    public Tensor<TNumber> Subtract<TNumber>(Tensor<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.Subtract(left, right, result);

        return result;
    }

    public Tensor<TNumber> Subtract<TNumber>(Tensor<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (!left.Shape.Equals(right.Shape))
        {
            throw new InvalidShapeException(
                $"The shape of the left tensor {left.Shape} must match the shape of the right tensor {right.Shape}.");
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.Subtract(left, right, result);

        return result;
    }

    public Scalar<TNumber> Multiply<TNumber>(Scalar<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Arithmetic.Multiply(left, right, result);

        return result;
    }

    public Vector<TNumber> Multiply<TNumber>(Scalar<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Vector<TNumber>(right.Length, backend);

        backend.Arithmetic.Multiply(left, right, result);

        return result;
    }

    public Matrix<TNumber> Multiply<TNumber>(Scalar<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.Multiply(left, right, result);

        return result;
    }

    public Tensor<TNumber> Multiply<TNumber>(Scalar<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Tensor<TNumber>(right.Shape, backend);

        backend.Arithmetic.Multiply(left, right, result);

        return result;
    }

    public Vector<TNumber> Multiply<TNumber>(Vector<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Vector<TNumber>(left.Length, backend);

        backend.Arithmetic.Multiply(left, right, result);

        return result;
    }

    public Matrix<TNumber> Multiply<TNumber>(Matrix<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.Multiply(left, right, result);

        return result;
    }

    public Tensor<TNumber> Multiply<TNumber>(Tensor<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.Multiply(left, right, result);

        return result;
    }

    public Tensor<TNumber> Multiply<TNumber>(Tensor<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Shape != right.Shape)
        {
            throw new InvalidShapeException(
                $"The shape of the left tensor {left.Shape} must match the shape of the right tensor {right.Shape}.");
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);

        var backend = left.Backend;
        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.Multiply(left, right, result);

        return result;
    }

    public Scalar<TNumber> Divide<TNumber>(Scalar<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Arithmetic.Divide(left, right, result);

        return result;
    }

    public Vector<TNumber> Divide<TNumber>(Scalar<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Vector<TNumber>(right.Length, backend);

        backend.Arithmetic.Divide(left, right, result);

        return result;
    }

    public Matrix<TNumber> Divide<TNumber>(Scalar<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.Divide(left, right, result);

        return result;
    }

    public Tensor<TNumber> Divide<TNumber>(Scalar<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Tensor<TNumber>(right.Shape, backend);

        backend.Arithmetic.Divide(left, right, result);

        return result;
    }

    public Tensor<TNumber> Divide<TNumber>(Vector<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Vector<TNumber>(left.Length, backend);

        backend.Arithmetic.Divide(left, right, result);

        return result;
    }

    public Tensor<TNumber> Divide<TNumber>(Matrix<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.Divide(left, right, result);

        return result;
    }

    public Tensor<TNumber> Divide<TNumber>(Tensor<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.Divide(left, right, result);

        return result;
    }

    public Scalar<TNumber> Negate<TNumber>(Scalar<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Arithmetic.Negate(value, result);

        return result;
    }

    public Vector<TNumber> Negate<TNumber>(Vector<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Vector<TNumber>(value.Length, backend);

        backend.Arithmetic.Negate(value, result);

        return result;
    }

    public Matrix<TNumber> Negate<TNumber>(Matrix<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Matrix<TNumber>(value.Rows, value.Columns, backend);

        backend.Arithmetic.Negate(value, result);

        return result;
    }

    public Tensor<TNumber> Negate<TNumber>(Tensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Tensor<TNumber>(value.Shape, backend);

        backend.Arithmetic.Negate(value, result);

        return result;
    }

    public Scalar<TNumber> Abs<TNumber>(Scalar<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Arithmetic.Abs(value, result);

        return result;
    }

    public Vector<TNumber> Abs<TNumber>(Vector<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Vector<TNumber>(value.Length, backend);

        backend.Arithmetic.Abs(value, result);

        return result;
    }

    public Matrix<TNumber> Abs<TNumber>(Matrix<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Matrix<TNumber>(value.Rows, value.Columns, backend);

        backend.Arithmetic.Abs(value, result);

        return result;
    }

    public Tensor<TNumber> Abs<TNumber>(Tensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Tensor<TNumber>(value.Shape, backend);

        backend.Arithmetic.Abs(value, result);

        return result;
    }
}