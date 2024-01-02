// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.Pointwise.Implementations;

internal class ArithmeticService : IArithmeticService
{
    private readonly IDeviceGuardService _guardService;

    public ArithmeticService(ITensorOperationServiceProvider provider)
    {
        _guardService = provider.GetDeviceGuardService();
    }

    public Scalar<TNumber> Add<TNumber>(
        Scalar<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Arithmetic.AddTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            1);

        return result;
    }

    public Vector<TNumber> Add<TNumber>(
        Scalar<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Vector<TNumber>(right.Length, backend);

        backend.Arithmetic.AddBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount,
            1);

        return result;
    }

    public Matrix<TNumber> Add<TNumber>(
        Scalar<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.AddBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount,
            1);

        return result;
    }

    public Tensor<TNumber> Add<TNumber>(
        Scalar<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Tensor<TNumber>(right.Shape, backend);

        backend.Arithmetic.AddBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount,
            1);

        return result;
    }

    public Vector<TNumber> Add<TNumber>(
        Vector<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Vector<TNumber>(left.Length, backend);

        backend.Arithmetic.AddTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount / right.Shape.ElementCount,
            right.Shape.ElementCount);

        return result;
    }

    public Vector<TNumber> Add<TNumber>(
        Vector<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Length != right.Length)
        {
            throw new InvalidShapeException($"Cannot add vectors of different lengths: {left.Length} and {right.Length}.");
        }

        var result = new Vector<TNumber>(left.Length, backend);

        backend.Arithmetic.AddTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Length);

        return result;
    }

    public Matrix<TNumber> Add<TNumber>(
        Vector<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Length != right.Columns)
        {
            throw new InvalidShapeException($"Cannot add vector of length {left.Length} to matrix with {right.Columns}.");
        }

        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.AddBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount / left.Length,
            left.Length);

        return result;
    }

    public Tensor<TNumber> Add<TNumber>(
        Vector<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Length != right.Shape[^1])
        {
            throw new InvalidShapeException($"Cannot add vector of length {left.Length} to tensor with shape {right.Shape}.");
        }

        var result = new Tensor<TNumber>(right.Shape, backend);

        backend.Arithmetic.AddBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount / left.Length,
            left.Length);

        return result;
    }

    public Matrix<TNumber> Add<TNumber>(
        Matrix<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.AddTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount,
            1);

        return result;
    }

    public Matrix<TNumber> Add<TNumber>(
        Matrix<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Columns != right.Length)
        {
            throw new InvalidShapeException($"Cannot add matrix with {left.Columns} columns to vector of length {right.Length}.");
        }

        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.AddTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount / right.Length,
            right.Length);

        return result;
    }

    public Matrix<TNumber> Add<TNumber>(
        Matrix<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Rows != right.Rows || left.Columns != right.Columns)
        {
            throw new InvalidShapeException($"Cannot add matrices with different shapes: {left.Shape} and {right.Shape}.");
        }

        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.AddTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount);

        return result;
    }

    public Tensor<TNumber> Add<TNumber>(
        Matrix<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Rows != right.Shape[^2] || left.Columns != right.Shape[^1])
        {
            throw new InvalidShapeException($"Cannot add matrix with shape {left.Shape} to tensor with shape {right.Shape}.");
        }

        var result = new Tensor<TNumber>(right.Shape, backend);

        backend.Arithmetic.AddBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount / left.Shape.ElementCount,
            left.Shape.ElementCount);

        return result;
    }

    public Tensor<TNumber> Add<TNumber>(
        Tensor<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.AddTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount,
            1);

        return result;
    }

    public Tensor<TNumber> Add<TNumber>(
        Tensor<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Shape[^1] != right.Length)
        {
            throw new InvalidShapeException($"Cannot add tensor with shape {left.Shape} to vector of length {right.Length}.");
        }

        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.AddTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount / right.Length,
            right.Length);

        return result;
    }

    public Tensor<TNumber> Add<TNumber>(
        Tensor<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Shape[^1] != right.Columns || left.Shape[^2] != right.Rows)
        {
            throw new InvalidShapeException($"Cannot add tensor with shape {left.Shape} to matrix with shape {right.Shape}.");
        }

        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.AddTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount / right.Shape.ElementCount,
            right.Shape.ElementCount);

        return result;
    }

    public Tensor<TNumber> Add<TNumber>(
        Tensor<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Shape == right.Shape)
        {
            var result = new Tensor<TNumber>(left.Shape, backend);

            backend.Arithmetic.AddTensorTensor(
                left.Memory,
                right.Memory,
                result.Memory,
                left.Shape.ElementCount);

            return result;
        }

        if (left.Shape.ElementCount > right.Shape.ElementCount)
        {
            for (var i = left.Shape.Rank - 1; i > right.Shape.Rank - 1; i--)
            {
                if (left.Shape[i] != right.Shape[i])
                {
                    throw new InvalidShapeException($"Cannot add tensors with different shapes: {left.Shape} and {right.Shape}.");
                }
            }

            var result = new Tensor<TNumber>(left.Shape, backend);

            backend.Arithmetic.AddTensorBroadcastTensor(
                left.Memory,
                right.Memory,
                result.Memory,
                right.Shape.ElementCount,
                left.Shape.ElementCount / right.Shape.ElementCount);

            return result;
        }
        else
        {
            for (var i = right.Shape.Rank - 1; i > left.Shape.Rank - 1; i--)
            {
                if (left.Shape[i] != right.Shape[i])
                {
                    throw new InvalidShapeException($"Cannot add tensors with different shapes: {left.Shape} and {right.Shape}.");
                }
            }

            var result = new Tensor<TNumber>(right.Shape, backend);

            backend.Arithmetic.AddTensorBroadcastTensor(
                left.Memory,
                right.Memory,
                result.Memory,
                left.Shape.ElementCount,
                right.Shape.ElementCount / left.Shape.ElementCount);

            return result;
        }
    }

    public ITensor<TNumber> Add<TNumber>(
        ITensor<TNumber> left,
        ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.IsScalar())
        {
            if (right.IsScalar())
            {
                return Add(left.ToScalar(), right.ToScalar());
            }

            if (right.IsVector())
            {
                return Add(left.ToScalar(), right.ToVector());
            }

            if (right.IsMatrix())
            {
                return Add(left.ToScalar(), right.ToMatrix());
            }

            return Add(left.ToScalar(), right.ToTensor());
        }

        if (left.IsVector())
        {
            if (right.IsScalar())
            {
                return Add(left.ToVector(), right.ToScalar());
            }

            if (right.IsVector())
            {
                return Add(left.ToVector(), right.ToVector());
            }

            if (right.IsMatrix())
            {
                return Add(left.ToVector(), right.ToMatrix());
            }

            return Add(left.ToVector(), right.ToTensor());
        }

        if (left.IsMatrix())
        {
            if (right.IsScalar())
            {
                return Add(left.ToMatrix(), right.ToScalar());
            }

            if (right.IsVector())
            {
                return Add(left.ToMatrix(), right.ToVector());
            }

            if (right.IsMatrix())
            {
                return Add(left.ToMatrix(), right.ToMatrix());
            }

            return Add(left.ToMatrix(), right.ToTensor());
        }

        if (left.IsTensor())
        {
            if (right.IsScalar())
            {
                return Add(left.ToTensor(), right.ToScalar());
            }

            if (right.IsVector())
            {
                return Add(left.ToTensor(), right.ToVector());
            }

            if (right.IsMatrix())
            {
                return Add(left.ToTensor(), right.ToMatrix());
            }

            return Add(left.ToTensor(), right.ToTensor());
        }

        throw new UnreachableException();
    }

    public Scalar<TNumber> Subtract<TNumber>(
        Scalar<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Scalar<TNumber>(backend);

        backend.Arithmetic.SubtractTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            1);

        return result;
    }

    public Vector<TNumber> Subtract<TNumber>(
        Scalar<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Vector<TNumber>(right.Length, backend);

        backend.Arithmetic.SubtractBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount,
            1);

        return result;
    }

    public Matrix<TNumber> Subtract<TNumber>(
        Scalar<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.SubtractBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount,
            1);

        return result;
    }

    public Tensor<TNumber> Subtract<TNumber>(
        Scalar<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Tensor<TNumber>(right.Shape, backend);

        backend.Arithmetic.SubtractBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount,
            1);

        return result;
    }

    public Vector<TNumber> Subtract<TNumber>(
        Vector<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Vector<TNumber>(left.Length, backend);

        backend.Arithmetic.SubtractTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount / right.Shape.ElementCount,
            right.Shape.ElementCount);

        return result;
    }

    public Vector<TNumber> Subtract<TNumber>(
        Vector<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Length != right.Length)
        {
            throw new InvalidShapeException(
                $"The length of the left vector ({left.Length}) " +
                $"does not match the length of the right vector ({right.Length}).");
        }

        var result = new Vector<TNumber>(left.Length, backend);

        backend.Arithmetic.SubtractTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Length);

        return result;
    }

    public Matrix<TNumber> Subtract<TNumber>(
        Vector<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Length != right.Columns)
        {
            throw new InvalidShapeException(
                $"The length of the left vector ({left.Length}) " +
                $"does not match the number of columns of the right matrix ({right.Columns}).");
        }

        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.SubtractBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount / left.Length,
            left.Length);

        return result;
    }

    public Tensor<TNumber> Subtract<TNumber>(
        Vector<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Length != right.Shape[^1])
        {
            throw new ArgumentException(
                $"The length of the left vector ({left.Length}) " +
                $"does not match the last dimension of the right tensor ({right.Shape[0]}).");
        }

        var result = new Tensor<TNumber>(right.Shape, backend);

        backend.Arithmetic.SubtractBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount / left.Length,
            left.Length);

        return result;
    }

    public Matrix<TNumber> Subtract<TNumber>(
        Matrix<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.SubtractTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount,
            1);

        return result;
    }

    public Matrix<TNumber> Subtract<TNumber>(
        Matrix<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Columns != right.Length)
        {
            throw new InvalidShapeException(
                $"The number of columns of the left matrix ({left.Columns}) " +
                $"does not match the length of the right vector ({right.Length}).");
        }

        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.SubtractTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount,
            left.Shape.ElementCount / right.Shape.ElementCount);

        return result;
    }

    public Matrix<TNumber> Subtract<TNumber>(
        Matrix<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);

        if (left.Rows != right.Rows || left.Columns != right.Columns)
        {
            throw new InvalidShapeException(
                $"The shape of the left matrix ({left.Rows}, {left.Columns}) " +
                $"does not match the shape of the right matrix ({right.Rows}, {right.Columns}).");
        }

        var backend = left.Backend;
        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.SubtractTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount);

        return result;
    }

    public Tensor<TNumber> Subtract<TNumber>(
        Matrix<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);

        if (left.Columns != right.Shape[^2] || left.Rows != right.Shape[^1])
        {
            throw new InvalidShapeException(
                $"The shape of the left matrix ({left.Rows}, {left.Columns}) " +
                $"does not match the last dimensions of the shape of the right tensor ({right.Shape[^2]}, {right.Shape[^1]}).");
        }

        var backend = left.Backend;
        var result = new Tensor<TNumber>(right.Shape, backend);

        backend.Arithmetic.SubtractBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount / left.Shape.ElementCount,
            left.Shape.ElementCount);

        return result;
    }

    public Tensor<TNumber> Subtract<TNumber>(
        Tensor<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.SubtractTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount,
            1);

        return result;
    }

    public Tensor<TNumber> Subtract<TNumber>(
        Tensor<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Shape[^1] != right.Length)
        {
            throw new InvalidShapeException(
                $"The last dimension of the left tensor ({left.Shape[^1]}) " +
                $"does not match the length of the right vector ({right.Length}).");
        }

        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.SubtractTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Length,
            left.Shape.ElementCount / right.Length);

        return result;
    }

    public Tensor<TNumber> Subtract<TNumber>(
        Tensor<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);

        if (left.Shape[^1] != right.Rows || left.Shape[^2] != right.Columns)
        {
            throw new InvalidShapeException(
                $"The last two dimensions of the left tensor ({left.Shape[^2]}, {left.Shape[^1]}) " +
                $"do not match the shape of the right matrix ({right.Rows}, {right.Columns}).");
        }

        var backend = left.Backend;
        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.SubtractTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount / right.Shape.ElementCount,
            right.Shape.ElementCount);

        return result;
    }

    public Tensor<TNumber> Subtract<TNumber>(
        Tensor<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Shape == right.Shape)
        {
            var result = new Tensor<TNumber>(left.Shape, backend);

            backend.Arithmetic.SubtractTensorTensor(
                left.Memory,
                right.Memory,
                result.Memory,
                left.Shape.ElementCount);

            return result;
        }

        if (left.Shape.ElementCount > right.Shape.ElementCount)
        {
            for (var i = left.Shape.Rank - 1; i > right.Shape.Rank - 1; i--)
            {
                if (left.Shape[i] != right.Shape[i])
                {
                    throw new InvalidShapeException($"Cannot add tensors with different shapes: {left.Shape} and {right.Shape}.");
                }
            }

            var result = new Tensor<TNumber>(left.Shape, backend);

            backend.Arithmetic.SubtractTensorBroadcastTensor(
                left.Memory,
                right.Memory,
                result.Memory,
                right.Shape.ElementCount,
                left.Shape.ElementCount / right.Shape.ElementCount);

            return result;
        }
        else
        {
            for (var i = right.Shape.Rank - 1; i > left.Shape.Rank - 1; i--)
            {
                if (left.Shape[i] != right.Shape[i])
                {
                    throw new InvalidShapeException($"Cannot add tensors with different shapes: {left.Shape} and {right.Shape}.");
                }
            }

            var result = new Tensor<TNumber>(right.Shape, backend);

            backend.Arithmetic.SubtractTensorBroadcastTensor(
                left.Memory,
                right.Memory,
                result.Memory,
                left.Shape.ElementCount,
                right.Shape.ElementCount / left.Shape.ElementCount);

            return result;
        }
    }

    public ITensor<TNumber> Subtract<TNumber>(
        ITensor<TNumber> left,
        ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.IsScalar())
        {
            if (right.IsScalar())
            {
                return Subtract(left.ToScalar(), right.ToScalar());
            }

            if (right.IsVector())
            {
                return Subtract(left.ToScalar(), right.ToVector());
            }

            if (right.IsMatrix())
            {
                return Subtract(left.ToScalar(), right.ToMatrix());
            }

            return Subtract(left.ToScalar(), right.ToTensor());
        }

        if (left.IsVector())
        {
            if (right.IsScalar())
            {
                return Subtract(left.ToVector(), right.ToScalar());
            }

            if (right.IsVector())
            {
                return Subtract(left.ToVector(), right.ToVector());
            }

            if (right.IsMatrix())
            {
                return Subtract(left.ToVector(), right.ToMatrix());
            }

            return Subtract(left.ToVector(), right.ToTensor());
        }

        if (left.IsMatrix())
        {
            if (right.IsScalar())
            {
                return Subtract(left.ToMatrix(), right.ToScalar());
            }

            if (right.IsVector())
            {
                return Subtract(left.ToMatrix(), right.ToVector());
            }

            if (right.IsMatrix())
            {
                return Subtract(left.ToMatrix(), right.ToMatrix());
            }

            return Subtract(left.ToMatrix(), right.ToTensor());
        }

        if (left.IsTensor())
        {
            if (right.IsScalar())
            {
                return Subtract(left.ToTensor(), right.ToScalar());
            }

            if (right.IsVector())
            {
                return Subtract(left.ToTensor(), right.ToVector());
            }

            if (right.IsMatrix())
            {
                return Subtract(left.ToTensor(), right.ToMatrix());
            }

            return Subtract(left.ToTensor(), right.ToTensor());
        }

        throw new UnreachableException();
    }

    public Scalar<TNumber> Multiply<TNumber>(
        Scalar<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Scalar<TNumber>(backend);

        backend.Arithmetic.MultiplyTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            1);

        return result;
    }

    public Vector<TNumber> Multiply<TNumber>(
        Scalar<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Vector<TNumber>(right.Length, backend);

        backend.Arithmetic.MultiplyBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount,
            1);
        return result;
    }

    public Matrix<TNumber> Multiply<TNumber>(
        Scalar<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.MultiplyBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount,
            1);

        return result;
    }

    public Tensor<TNumber> Multiply<TNumber>(
        Scalar<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Tensor<TNumber>(right.Shape, backend);

        backend.Arithmetic.MultiplyBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount,
            1);

        return result;
    }

    public Vector<TNumber> Multiply<TNumber>(
        Vector<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Vector<TNumber>(left.Length, backend);

        backend.Arithmetic.MultiplyTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount / right.Shape.ElementCount,
            right.Shape.ElementCount);

        return result;
    }

    public Vector<TNumber> Multiply<TNumber>(
        Vector<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Length != right.Length)
        {
            throw new InvalidShapeException(
                $"The length of the left vector ({left.Shape}) " +
                $"does not match the length of the right vector ({right.Shape}).");
        }

        var result = new Vector<TNumber>(left.Length, backend);

        backend.Arithmetic.MultiplyTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Length);

        return result;
    }

    public Matrix<TNumber> Multiply<TNumber>(
        Vector<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Length != right.Columns)
        {
            throw new InvalidShapeException(
                $"The length of the left vector ({left.Shape}) " +
                $"does not match the number of rows of the right matrix ({right.Shape}).");
        }

        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.MultiplyBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Length,
            right.Rows);

        return result;
    }

    public Tensor<TNumber> Multiply<TNumber>(
        Vector<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Length != right.Shape[^1])
        {
            throw new InvalidShapeException($"The length of the left vector ({left.Shape}) the last dimension of the right tensor ({right.Shape}).");
        }

        var result = new Tensor<TNumber>(right.Shape, backend);

        backend.Arithmetic.MultiplyBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount / left.Length,
            left.Length);
        return result;
    }

    public Matrix<TNumber> Multiply<TNumber>(
        Matrix<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.MultiplyTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount,
            1);

        return result;
    }

    public Matrix<TNumber> Multiply<TNumber>(
        Matrix<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Columns != right.Length)
        {
            throw new InvalidShapeException(
                $"The number of columns of the left matrix ({left.Shape}) " +
                $"does not match the length of the right vector ({right.Shape}).");
        }

        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.MultiplyTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount / right.Length,
            right.Length);

        return result;
    }

    public Matrix<TNumber> Multiply<TNumber>(
        Matrix<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Shape != right.Shape)
        {
            throw new InvalidShapeException(
                $"The shape of the left matrix ({left.Shape}) " +
                $"does not match the shape of the right matrix ({right.Shape}).");
        }

        var result = new Matrix<TNumber>(left.Rows, right.Columns, backend);

        backend.Arithmetic.MultiplyTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount);

        return result;
    }

    public Tensor<TNumber> Multiply<TNumber>(
        Matrix<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Rows != right.Shape[^2] || left.Columns != right.Shape[^1])
        {
            throw new InvalidShapeException(
                $"The shape of the left matrix ({left.Shape}) " +
                $"does not match the last and second to last dimensions of the right tensor ({right.Shape}).");
        }

        var result = new Tensor<TNumber>(right.Shape, backend);

        backend.Arithmetic.MultiplyBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount / left.Shape.ElementCount,
            left.Shape.ElementCount);

        return result;
    }

    public Tensor<TNumber> Multiply<TNumber>(
        Tensor<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.MultiplyTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount,
            1);

        return result;
    }

    public Tensor<TNumber> Multiply<TNumber>(
        Tensor<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Shape[^1] != right.Length)
        {
            throw new InvalidShapeException(
                $"The last dimension of the left tensor ({left.Shape}) " +
                $"does not match the length of the right vector ({right.Shape}).");
        }

        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.MultiplyTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount / right.Length,
            right.Length);

        return result;
    }

    public Tensor<TNumber> Multiply<TNumber>(
        Tensor<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Shape[^2] != right.Rows || left.Shape[^1] != right.Columns)
        {
            throw new InvalidShapeException(
                $"The last two dimension of the left tensor ({left.Shape}) " +
                $"does not match the shape of the right matrix ({right.Shape}).");
        }

        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.MultiplyTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount / right.Shape.ElementCount,
            right.Shape.ElementCount);

        return result;
    }

    public Tensor<TNumber> Multiply<TNumber>(
        Tensor<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Shape == right.Shape)
        {
            var result = new Tensor<TNumber>(left.Shape, backend);

            backend.Arithmetic.MultiplyTensorTensor(
                left.Memory,
                right.Memory,
                result.Memory,
                left.Shape.ElementCount);

            return result;
        }

        if (left.Shape.ElementCount > right.Shape.ElementCount)
        {
            for (var i = left.Shape.Rank - 1; i > right.Shape.Rank - 1; i--)
            {
                if (left.Shape[i] != right.Shape[i])
                {
                    throw new InvalidShapeException($"Cannot add tensors with different shapes: {left.Shape} and {right.Shape}.");
                }
            }

            var result = new Tensor<TNumber>(left.Shape, backend);

            backend.Arithmetic.MultiplyTensorBroadcastTensor(
                left.Memory,
                right.Memory,
                result.Memory,
                right.Shape.ElementCount,
                left.Shape.ElementCount / right.Shape.ElementCount);

            return result;
        }
        else
        {
            for (var i = right.Shape.Rank - 1; i > left.Shape.Rank - 1; i--)
            {
                if (left.Shape[i] != right.Shape[i])
                {
                    throw new InvalidShapeException($"Cannot add tensors with different shapes: {left.Shape} and {right.Shape}.");
                }
            }

            var result = new Tensor<TNumber>(right.Shape, backend);

            backend.Arithmetic.MultiplyTensorBroadcastTensor(
                left.Memory,
                right.Memory,
                result.Memory,
                left.Shape.ElementCount,
                right.Shape.ElementCount / left.Shape.ElementCount);

            return result;
        }
    }

    public Scalar<TNumber> Divide<TNumber>(
        Scalar<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Scalar<TNumber>(backend);

        backend.Arithmetic.DivideTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            1);

        return result;
    }

    public Vector<TNumber> Divide<TNumber>(
        Scalar<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Vector<TNumber>(right.Length, backend);

        backend.Arithmetic.DivideBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount,
            1);

        return result;
    }

    public Matrix<TNumber> Divide<TNumber>(
        Scalar<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.DivideBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount,
            1);

        return result;
    }

    public Tensor<TNumber> Divide<TNumber>(
        Scalar<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Tensor<TNumber>(right.Shape, backend);

        backend.Arithmetic.DivideBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount,
            1);

        return result;
    }

    public Vector<TNumber> Divide<TNumber>(
        Vector<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Vector<TNumber>(left.Length, backend);

        backend.Arithmetic.DivideTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Length,
            1);

        return result;
    }

    public Vector<TNumber> Divide<TNumber>(
        Vector<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Length != right.Length)
        {
            throw new InvalidShapeException($"Cannot divide vectors with different lengths: {left.Shape} and {right.Shape}.");
        }

        var result = new Vector<TNumber>(left.Length, backend);

        backend.Arithmetic.DivideTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Length);

        return result;
    }

    public Matrix<TNumber> Divide<TNumber>(
        Vector<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Length != right.Columns)
        {
            throw new InvalidShapeException($"Cannot divide vector with length {left.Shape} by matrix with shape {right.Shape}.");
        }

        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.DivideBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount / left.Length,
            left.Length);

        return result;
    }

    public Tensor<TNumber> Divide<TNumber>(
        Vector<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Length != right.Shape[^1])
        {
            throw new InvalidShapeException($"Cannot divide vector with length {left.Shape} by tensor with shape {right.Shape}.");
        }

        var result = new Tensor<TNumber>(right.Shape, backend);

        backend.Arithmetic.DivideBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Shape.ElementCount / left.Length,
            left.Length);

        return result;
    }

    public Matrix<TNumber> Divide<TNumber>(
        Matrix<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.DivideTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount,
            1);

        return result;
    }

    public Matrix<TNumber> Divide<TNumber>(
        Matrix<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Columns != right.Length)
        {
            throw new InvalidShapeException($"Cannot divide matrix with shape {left.Shape} by vector with length {right.Shape}.");
        }

        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.DivideTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            right.Length,
            left.Rows);

        return result;
    }

    public Matrix<TNumber> Divide<TNumber>(
        Matrix<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);

        var backend = left.Backend;

        if (left.Rows != right.Rows || left.Columns != right.Columns)
        {
            throw new InvalidShapeException($"Cannot divide matrices with different shapes: {left.Shape} and {right.Shape}.");
        }

        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.DivideTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Rows * left.Columns);

        return result;
    }

    public Tensor<TNumber> Divide<TNumber>(
        Matrix<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);

        var backend = left.Backend;

        if (left.Rows != right.Shape[^1] || left.Columns != right.Shape[^1])
        {
            throw new InvalidShapeException($"Cannot divide matrix with shape {left.Shape} by tensor with shape {right.Shape}.");
        }

        var result = new Tensor<TNumber>(right.Shape, backend);

        backend.Arithmetic.DivideBroadcastTensorTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Columns,
            right.Shape.ElementCount / left.Columns);

        return result;
    }

    public Tensor<TNumber> Divide<TNumber>(
        Tensor<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.DivideTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount,
            1);

        return result;
    }

    public Tensor<TNumber> Divide<TNumber>(
        Tensor<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Shape[^1] != right.Length)
        {
            throw new InvalidShapeException($"Cannot divide tensor with shape {left.Shape} by vector with length {right.Length}.");
        }

        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.DivideTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount / right.Length,
            right.Length);

        return result;
    }

    public Tensor<TNumber> Divide<TNumber>(
        Tensor<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Shape[^2] != right.Rows || left.Shape[^1] != right.Columns)
        {
            throw new InvalidShapeException($"Cannot divide tensor with shape {left.Shape} by matrix with shape {right.Shape}.");
        }

        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.DivideTensorBroadcastTensor(
            left.Memory,
            right.Memory,
            result.Memory,
            left.Shape.ElementCount / right.Shape.ElementCount,
            right.Shape.ElementCount);

        return result;
    }

    public Tensor<TNumber> Divide<TNumber>(
        Tensor<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        if (left.Shape == right.Shape)
        {
            var result = new Tensor<TNumber>(left.Shape, backend);

            backend.Arithmetic.DivideTensorTensor(
                left.Memory,
                right.Memory,
                result.Memory,
                left.Shape.ElementCount);

            return result;
        }

        if (left.Shape.ElementCount > right.Shape.ElementCount)
        {
            for (var i = left.Shape.Rank - 1; i > right.Shape.Rank - 1; i--)
            {
                if (left.Shape[i] != right.Shape[i])
                {
                    throw new InvalidShapeException($"Cannot add tensors with different shapes: {left.Shape} and {right.Shape}.");
                }
            }

            var result = new Tensor<TNumber>(left.Shape, backend);

            backend.Arithmetic.DivideTensorBroadcastTensor(
                left.Memory,
                right.Memory,
                result.Memory,
                right.Shape.ElementCount,
                left.Shape.ElementCount / right.Shape.ElementCount);

            return result;
        }
        else
        {
            for (var i = right.Shape.Rank - 1; i > left.Shape.Rank - 1; i--)
            {
                if (left.Shape[i] != right.Shape[i])
                {
                    throw new InvalidShapeException($"Cannot add tensors with different shapes: {left.Shape} and {right.Shape}.");
                }
            }

            var result = new Tensor<TNumber>(right.Shape, backend);

            backend.Arithmetic.DivideTensorBroadcastTensor(
                left.Memory,
                right.Memory,
                result.Memory,
                left.Shape.ElementCount,
                right.Shape.ElementCount / left.Shape.ElementCount);

            return result;
        }
    }

    public Scalar<TNumber> Negate<TNumber>(Scalar<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;

        if (!GenericMath.IsSigned<TNumber>())
        {
            var newMemoryBlock = value.Memory.Copy();

            return new Scalar<TNumber>(newMemoryBlock, backend);
        }

        var result = new Scalar<TNumber>(backend);

        backend.Arithmetic.Negate(
            value.Memory,
            result.Memory,
            value.Shape.ElementCount);

        return result;
    }

    public Vector<TNumber> Negate<TNumber>(Vector<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;

        if (!GenericMath.IsSigned<TNumber>())
        {
            var newMemoryBlock = value.Memory.Copy();

            return new Vector<TNumber>(value.Length, newMemoryBlock, backend);
        }

        var result = new Vector<TNumber>(value.Length, backend);

        backend.Arithmetic.Negate(
            value.Memory,
            result.Memory,
            value.Shape.ElementCount);

        return result;
    }

    public Matrix<TNumber> Negate<TNumber>(Matrix<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;

        if (!GenericMath.IsSigned<TNumber>())
        {
            var newMemoryBlock = value.Memory.Copy();

            return new Matrix<TNumber>(value.Rows, value.Columns, newMemoryBlock, backend);
        }

        var result = new Matrix<TNumber>(value.Rows, value.Columns, backend);

        backend.Arithmetic.Negate(
            value.Memory,
            result.Memory,
            value.Shape.ElementCount);

        return result;
    }

    public Tensor<TNumber> Negate<TNumber>(Tensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;

        if (!GenericMath.IsSigned<TNumber>())
        {
            var newMemoryBlock = value.Memory.Copy();

            return new Tensor<TNumber>(newMemoryBlock, value.Shape, backend);
        }

        var result = new Tensor<TNumber>(value.Shape, backend);

        backend.Arithmetic.Negate(
            value.Memory,
            result.Memory,
            value.Shape.ElementCount);

        return result;
    }

    public Scalar<TNumber> Abs<TNumber>(Scalar<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Scalar<TNumber>(backend);

        backend.Arithmetic.Abs(
            value.Memory,
            result.Memory,
            1);

        return result;
    }

    public Vector<TNumber> Abs<TNumber>(Vector<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Vector<TNumber>(value.Length, backend);

        backend.Arithmetic.Abs(
            value.Memory,
            result.Memory,
            value.Length);

        return result;
    }

    public Matrix<TNumber> Abs<TNumber>(Matrix<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Matrix<TNumber>(value.Rows, value.Columns, backend);

        backend.Arithmetic.Abs(
            value.Memory,
            result.Memory,
            value.Shape.ElementCount);

        return result;
    }

    public Tensor<TNumber> Abs<TNumber>(Tensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = value.Backend;
        var result = new Tensor<TNumber>(value.Shape, backend);

        backend.Arithmetic.Abs(
            value.Memory,
            result.Memory,
            value.Shape.ElementCount);

        return result;
    }

    public ITensor<TNumber> Multiply<TNumber>(
        ITensor<TNumber> left,
        ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.IsScalar())
        {
            if (right.IsScalar())
            {
                return Multiply(left.ToScalar(), right.ToScalar());
            }

            if (right.IsVector())
            {
                return Multiply(left.ToScalar(), right.ToVector());
            }

            if (right.IsMatrix())
            {
                return Multiply(left.ToScalar(), right.ToMatrix());
            }

            return Multiply(left.ToScalar(), right.ToTensor());
        }

        if (left.IsVector())
        {
            if (right.IsScalar())
            {
                return Multiply(left.ToVector(), right.ToScalar());
            }

            if (right.IsVector())
            {
                return Multiply(left.ToVector(), right.ToVector());
            }

            if (right.IsMatrix())
            {
                return Multiply(left.ToVector(), right.ToMatrix());
            }

            return Multiply(left.ToVector(), right.ToTensor());
        }

        if (left.IsMatrix())
        {
            if (right.IsScalar())
            {
                return Multiply(left.ToMatrix(), right.ToScalar());
            }

            if (right.IsVector())
            {
                return Multiply(left.ToMatrix(), right.ToVector());
            }

            if (right.IsMatrix())
            {
                return Multiply(left.ToMatrix(), right.ToMatrix());
            }

            return Multiply(left.ToMatrix(), right.ToTensor());
        }

        if (left.IsTensor())
        {
            if (right.IsScalar())
            {
                return Multiply(left.ToTensor(), right.ToScalar());
            }

            if (right.IsVector())
            {
                return Multiply(left.ToTensor(), right.ToVector());
            }

            if (right.IsMatrix())
            {
                return Multiply(left.ToTensor(), right.ToMatrix());
            }

            return Multiply(left.ToTensor(), right.ToTensor());
        }

        throw new UnreachableException();
    }

    public ITensor<TNumber> Divide<TNumber>(
        ITensor<TNumber> left,
        ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.IsScalar())
        {
            if (right.IsScalar())
            {
                return Divide(left.ToScalar(), right.ToScalar());
            }

            if (right.IsVector())
            {
                return Divide(left.ToScalar(), right.ToVector());
            }

            if (right.IsMatrix())
            {
                return Divide(left.ToScalar(), right.ToMatrix());
            }

            return Divide(left.ToScalar(), right.ToTensor());
        }

        if (left.IsVector())
        {
            if (right.IsScalar())
            {
                return Divide(left.ToVector(), right.ToScalar());
            }

            if (right.IsVector())
            {
                return Divide(left.ToVector(), right.ToVector());
            }

            if (right.IsMatrix())
            {
                return Divide(left.ToVector(), right.ToMatrix());
            }

            return Divide(left.ToVector(), right.ToTensor());
        }

        if (left.IsMatrix())
        {
            if (right.IsScalar())
            {
                return Divide(left.ToMatrix(), right.ToScalar());
            }

            if (right.IsVector())
            {
                return Divide(left.ToMatrix(), right.ToVector());
            }

            if (right.IsMatrix())
            {
                return Divide(left.ToMatrix(), right.ToMatrix());
            }

            return Divide(left.ToMatrix(), right.ToTensor());
        }

        if (left.IsTensor())
        {
            if (right.IsScalar())
            {
                return Divide(left.ToTensor(), right.ToScalar());
            }

            if (right.IsVector())
            {
                return Divide(left.ToTensor(), right.ToVector());
            }

            if (right.IsMatrix())
            {
                return Divide(left.ToTensor(), right.ToMatrix());
            }

            return Divide(left.ToTensor(), right.ToTensor());
        }

        throw new UnreachableException();
    }

    public ITensor<TNumber> Sqrt<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, IRootFunctions<TNumber>, INumber<TNumber>
    {
        var backend = tensor.Backend;
        var result = new Tensor<TNumber>(tensor.Shape, backend);

        backend.Arithmetic.Sqrt(
            tensor.Memory,
            result.Memory,
            tensor.Shape.ElementCount);

        return result;
    }
}