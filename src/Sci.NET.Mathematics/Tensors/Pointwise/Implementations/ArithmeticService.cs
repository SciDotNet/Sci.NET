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
            throw new InvalidShapeException($"Cannot add shapes {left.Shape} and {right.Shape}.");
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
            if (!left.CanBroadcastTo(right.Shape))
            {
                throw new InvalidShapeException($"Cannot add shapes {left.Shape} and {right.Shape}.");
            }

            using var leftBroadcast = left.Broadcast(right.Shape);
            using var resultTensor = Add(leftBroadcast, right);
            return resultTensor.ToMatrix();
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.Add(left, right, result);

        return result;
    }

    public Tensor<TNumber> Add<TNumber>(Vector<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (_broadcastService.CanBroadcastTo(left.Shape, right.Shape))
        {
            return Add(left.Broadcast(right.Shape).ToTensor(), right);
        }

        throw new InvalidShapeException($"Cannot broadcast shapes {left} and {right}.");
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
            if (!right.CanBroadcastTo(left.Shape))
            {
                throw new InvalidShapeException($"Cannot add shapes {left.Shape} and {right.Shape}.");
            }

            using var rightBroadcast = right.Broadcast(left.Shape).ToMatrix();

            return left.Add(rightBroadcast);
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
            throw new InvalidShapeException($"Cannot add shapes {left.Shape} and {right.Shape}.");
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.Add(left, right, result);

        return result;
    }

    public Tensor<TNumber> Add<TNumber>(Matrix<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (_broadcastService.CanBroadcastTo(left.Shape, right.Shape))
        {
            return Add(left.Broadcast(right.Shape).ToTensor(), right);
        }

        throw new InvalidShapeException($"Cannot broadcast shapes {left} and {right}.");
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

    public Tensor<TNumber> Add<TNumber>(Tensor<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (_broadcastService.CanBroadcastTo(right.Shape, left.Shape))
        {
            return Add(left, left.Broadcast(right.Shape).ToTensor());
        }

        throw new InvalidShapeException($"Cannot broadcast shapes {left} and {right}.");
    }

    public Tensor<TNumber> Add<TNumber>(Tensor<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (_broadcastService.CanBroadcastTo(right.Shape, left.Shape))
        {
            return Add(left, left.Broadcast(right.Shape).ToTensor());
        }

        throw new InvalidShapeException($"Cannot broadcast shapes {left} and {right}.");
    }

    public Tensor<TNumber> Add<TNumber>(Tensor<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (!left.Shape.Equals(right.Shape))
        {
            throw new InvalidShapeException($"Cannot add shapes {left.Shape} and {right.Shape}.");
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
            if (_broadcastService.CanBroadcastTo(left.Shape, right.Shape))
            {
                using var broadcast = left.Broadcast(right.Shape);
                using var resultVector = Subtract(broadcast, right);
                return resultVector.ToVector();
            }

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
            if (_broadcastService.CanBroadcastTo(left.Shape, right.Shape))
            {
                using var broadcast = left.Broadcast(right.Shape);
                return broadcast.ToMatrix();
            }

            throw new InvalidShapeException(
                $"The length of the left vector {left.Shape} must match the length of the right vector {right.Shape}.");
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.Subtract(left, right, result);

        return result;
    }

    public Tensor<TNumber> Subtract<TNumber>(Vector<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (_broadcastService.CanBroadcastTo(left.Shape, right.Shape))
        {
            return Subtract(left.Broadcast(right.Shape).ToTensor(), right);
        }

        throw new InvalidShapeException($"Cannot broadcast shapes {left} and {right}.");
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

    public Tensor<TNumber> Subtract<TNumber>(Matrix<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (_broadcastService.CanBroadcastTo(left.Shape, right.Shape))
        {
            return Subtract(left.Broadcast(right.Shape).ToTensor(), right);
        }

        throw new InvalidShapeException($"Cannot broadcast shapes {left} and {right}.");
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

    public Tensor<TNumber> Subtract<TNumber>(Tensor<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (_broadcastService.CanBroadcastTo(right.Shape, left.Shape))
        {
            return Subtract(left, left.Broadcast(right.Shape).ToTensor());
        }

        throw new InvalidShapeException($"Cannot broadcast shapes {left} and {right}.");
    }

    public Tensor<TNumber> Subtract<TNumber>(Tensor<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (_broadcastService.CanBroadcastTo(right.Shape, left.Shape))
        {
            return Subtract(left, left.Broadcast(right.Shape).ToTensor());
        }

        throw new InvalidShapeException($"Cannot broadcast shapes {left} and {right}.");
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

    public ITensor<TNumber> Subtract<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
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

    public Vector<TNumber> Multiply<TNumber>(Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Length != right.Length)
        {
            throw new InvalidShapeException($"Cannot multiply shapes {left.Shape} and {right.Shape}.");
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Vector<TNumber>(left.Length, backend);

        backend.Arithmetic.Multiply(left, right, result);

        return result;
    }

    public Matrix<TNumber> Multiply<TNumber>(Vector<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Length != right.Rows)
        {
            if (!left.CanBroadcastTo(right.Shape))
            {
                throw new InvalidShapeException($"Cannot multiply shapes {left.Shape} and {right.Shape}.");
            }

            using var leftBroadcast = left.Broadcast(right.Shape);

            return leftBroadcast.Multiply(right).ToMatrix();
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.Multiply(left, right, result);

        return result;
    }

    public Tensor<TNumber> Multiply<TNumber>(Vector<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (_broadcastService.CanBroadcastTo(left.Shape, right.Shape))
        {
            using var broadcast = left.Broadcast(right.Shape);
            using var result = Multiply(broadcast, right);
            return result.ToTensor();
        }

        throw new InvalidShapeException($"Cannot broadcast shapes {left} and {right}.");
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

    public Matrix<TNumber> Multiply<TNumber>(Matrix<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (_broadcastService.CanBroadcastTo(left.Shape, right.Shape))
        {
            using var broadcast = left.Broadcast(right.Shape);
            using var result = Multiply(broadcast, right);
            return result.ToMatrix();
        }

        throw new InvalidShapeException($"Cannot broadcast shapes {left} and {right}.");
    }

    public Matrix<TNumber> Multiply<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Rows != right.Rows || left.Columns != right.Columns)
        {
            throw new InvalidShapeException($"Cannot multiply shapes {left.Shape} and {right.Shape}.");
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;

        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.Multiply(left, right, result);

        return result;
    }

    public Matrix<TNumber> Multiply<TNumber>(Matrix<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (_broadcastService.CanBroadcastTo(left.Shape, right.Shape))
        {
            using var broadcast = left.Broadcast(right.Shape);
            using var result = Multiply(broadcast, right);
            return result.ToMatrix();
        }

        throw new InvalidShapeException($"Cannot broadcast shapes {left} and {right}.");
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

    public Tensor<TNumber> Multiply<TNumber>(Tensor<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (!right.CanBroadcastTo(left.Shape))
        {
            throw new InvalidShapeException($"Cannot multiply shapes {left.Shape} and {right.Shape}.");
        }

        using var rightBroadcast = right.Broadcast(left.Shape);
        using var result = Multiply(left, rightBroadcast);

        return result.ToTensor();
    }

    public Tensor<TNumber> Multiply<TNumber>(Tensor<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (!right.CanBroadcastTo(left.Shape))
        {
            throw new InvalidShapeException($"Cannot multiply shapes {left.Shape} and {right.Shape}.");
        }

        using var rightBroadcast = right.Broadcast(left.Shape);
        using var result = Multiply(left, rightBroadcast);

        return result.ToTensor();
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

    public Vector<TNumber> Divide<TNumber>(Vector<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Vector<TNumber>(left.Length, backend);

        backend.Arithmetic.Divide(left, right, result);

        return result;
    }

    public Vector<TNumber> Divide<TNumber>(Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Length != right.Length)
        {
            throw new InvalidShapeException($"Cannot divide shapes {left.Shape} and {right.Shape}.");
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);

        var backend = left.Backend;
        var result = new Vector<TNumber>(left.Length, backend);

        backend.Arithmetic.Divide(left, right, result);

        return result;
    }

    public Matrix<TNumber> Divide<TNumber>(Vector<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Length != right.Rows)
        {
            if (!left.CanBroadcastTo(right.Shape))
            {
                throw new InvalidShapeException($"Cannot divide shapes {left.Shape} and {right.Shape}.");
            }

            using var leftBroadcast = left.Broadcast(right.Shape);
            using var resultTensor = Divide(leftBroadcast, right);

            return resultTensor.ToMatrix();
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(right.Rows, right.Columns, backend);

        backend.Arithmetic.Divide(left, right, result);

        return result;
    }

    public Tensor<TNumber> Divide<TNumber>(Vector<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (_broadcastService.CanBroadcastTo(right.Shape, left.Shape))
        {
            using var broadcast = left.Broadcast(left.Shape);
            using var result = Divide(left, broadcast);
            return result.ToTensor();
        }

        throw new InvalidShapeException($"Cannot broadcast shapes {left} and {right}.");
    }

    public Matrix<TNumber> Divide<TNumber>(Matrix<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.Divide(left, right, result);

        return result;
    }

    public Matrix<TNumber> Divide<TNumber>(Matrix<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (!right.CanBroadcastTo(left.Shape))
        {
            throw new InvalidShapeException($"Cannot divide shapes {left.Shape} and {right.Shape}.");
        }

        using var broadcast = right.Broadcast(left.Shape);
        using var result = Divide(left, broadcast);

        return result.ToMatrix();
    }

    public Matrix<TNumber> Divide<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Rows != right.Rows || left.Columns != right.Columns)
        {
            throw new InvalidShapeException($"Cannot divide shapes {left.Shape} and {right.Shape}.");
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Matrix<TNumber>(left.Rows, left.Columns, backend);

        backend.Arithmetic.Divide(left, right, result);

        return result;
    }

    public Tensor<TNumber> Divide<TNumber>(Matrix<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (_broadcastService.CanBroadcastTo(right.Shape, left.Shape))
        {
            using var broadcast = left.Broadcast(left.Shape);
            using var result = Divide(left, broadcast);
            return result.ToTensor();
        }

        throw new InvalidShapeException($"Cannot broadcast shapes {left} and {right}.");
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

    public Tensor<TNumber> Divide<TNumber>(Tensor<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Shape[0] != right.Length)
        {
            if (!right.CanBroadcastTo(left.Shape))
            {
                throw new InvalidShapeException($"Cannot divide shapes {left.Shape} and {right.Shape}.");
            }

            using var broadcast = right.Broadcast(left.Shape);
            using var resultTensor = Divide(left, broadcast);

            return resultTensor.ToTensor();
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.Divide(left, right, result);

        return result;
    }

    public Tensor<TNumber> Divide<TNumber>(Tensor<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Shape[0] != right.Rows)
        {
            if (!right.CanBroadcastTo(left.Shape))
            {
                throw new InvalidShapeException($"Cannot divide shapes {left.Shape} and {right.Shape}.");
            }

            using var broadcast = right.Broadcast(left.Shape);
            using var resultTensor = Divide(left, broadcast);

            return resultTensor.ToTensor();
        }

        _guardService.GuardBinaryOperation(left.Device, right.Device);
        var backend = left.Backend;
        var result = new Tensor<TNumber>(left.Shape, backend);

        backend.Arithmetic.Divide(left, right, result);

        return result;
    }

    public Tensor<TNumber> Divide<TNumber>(Tensor<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Shape != right.Shape)
        {
            if (!right.CanBroadcastTo(left.Shape))
            {
                throw new InvalidShapeException($"Cannot divide shapes {left.Shape} and {right.Shape}.");
            }

            using var broadcast = right.Broadcast(left.Shape);
            using var resultTensor = Divide(left, broadcast);

            return resultTensor.ToTensor();
        }

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

    public ITensor<TNumber> Multiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
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

    public ITensor<TNumber> Divide<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
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

        backend.Arithmetic.Sqrt(tensor, result);

        return result;
    }
}