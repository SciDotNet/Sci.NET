// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.Arithmetic.Implementations;

internal class ArithmeticService : IArithmeticService
{
    private readonly IDeviceGuardService _guardService;

    public ArithmeticService(ITensorOperationServiceProvider provider)
    {
        _guardService = provider.GetDeviceGuardService();
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
            throw new InvalidShapeException(
                "The length of the left vector ({0}) must match the length of the right vector ({1}).",
                left.Shape,
                right.Shape);
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
            throw new InvalidShapeException(
                "The length of the left vector ({0}) must match the length of the right vector ({1}).",
                left.Shape,
                right.Shape);
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
            throw new InvalidShapeException(
                "The length of the left vector ({0}) must match the length of the right vector ({1}).",
                left.Shape,
                right.Shape);
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
            throw new InvalidShapeException(
                "The shape of the left matrix ({0}) must match the shape of the right matrix ({1}).",
                left.Shape,
                right.Shape);
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
            throw new InvalidShapeException(
                "The shape of the left tensor ({0}) must match the shape of the right tensor ({1}).",
                left.Shape,
                right.Shape);
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.Add(left, right, result);

        return result;
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
                "The length of the left vector ({0}) must match the length of the right vector ({1}).",
                left.Shape,
                right.Shape);
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
                "The length of the left vector ({0}) must match the length of the right vector ({1}).",
                left.Shape,
                right.Shape);
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
                "The length of the left vector ({0}) must match the length of the right vector ({1}).",
                left.Shape,
                right.Shape);
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
                "The shape of the left matrix ({0}) must match the shape of the right matrix ({1}).",
                left.Shape,
                right.Shape);
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
                "The shape of the left tensor ({0}) must match the shape of the right tensor ({1}).",
                left.Shape,
                right.Shape);
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.Subtract(left, right, result);

        return result;
    }

    public ITensor<TNumber> Multiply<TNumber>(Scalar<TNumber> left, Scalar<TNumber> right)
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

    public ITensor<TNumber> Divide<TNumber>(Scalar<TNumber> left, Scalar<TNumber> right)
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
}