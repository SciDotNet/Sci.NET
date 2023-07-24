// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// An interface for a rank-N tensor, which is an immutable N-Dimensional array.
/// </summary>
/// <typeparam name="TNumber">The type of the numbers stored in the <see cref="ITensor{TNumber}"/>.</typeparam>
[PublicAPI]
public interface ITensor<TNumber> : ITensorLocalityOperations, IDisposable
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Gets the shape of the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    public Shape Shape { get; }

    /// <summary>
    /// Gets the handle to the memory of the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    public IMemoryBlock<TNumber> Handle { get; }

    /// <summary>
    /// Gets the backend of the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    public ITensorBackend Backend { get; }

    /// <summary>
    /// Gets the device used for the current <see cref="ITensor{TNumber}"/>.
    /// </summary>
    public IDevice Device { get; }

    /// <summary>
    /// Gets the debugger display object.
    /// </summary>
    [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
    private protected Array DebuggerDisplayObject => ToArray();

    /// <summary>
    /// Gets the slice of the <see cref="ITensor{TNumber}"/> at the specified indices.
    /// </summary>
    /// <param name="indices">The indices of the <see cref="ITensor{TNumber}"/> to slice.</param>
#pragma warning disable CA1043
    public ITensor<TNumber> this[params int[] indices] { get; }

#pragma warning restore CA1043

#pragma warning disable CS1591
    public static ITensor<TNumber> operator +(ITensor<TNumber> left, ITensor<TNumber> right)
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

    public static ITensor<TNumber> operator -(ITensor<TNumber> left, ITensor<TNumber> right)
    {
        if (left.IsScalar())
        {
            var leftScalar = left.AsScalar();

            if (right.IsScalar())
            {
                return leftScalar.Subtract(right.AsScalar());
            }

            if (right.IsVector())
            {
                return leftScalar.Subtract(right.AsVector());
            }

            if (right.IsMatrix())
            {
                return leftScalar.Subtract(right.AsMatrix());
            }

            return leftScalar.Subtract(right.AsTensor());
        }

        if (left.IsVector())
        {
            var leftVector = left.AsVector();

            if (right.IsScalar())
            {
                return leftVector.Subtract(right.AsScalar());
            }

            if (right.IsVector())
            {
                return leftVector.Subtract(right.AsVector());
            }

            if (right.IsMatrix())
            {
                return leftVector.Subtract(right.AsMatrix());
            }

            return leftVector.Subtract(right.AsTensor());
        }

        if (left.IsMatrix())
        {
            var leftMatrix = left.AsMatrix();

            if (right.IsScalar())
            {
                return leftMatrix.Subtract(right.AsScalar());
            }

            if (right.IsVector())
            {
                return leftMatrix.Subtract(right.AsVector());
            }

            if (right.IsMatrix())
            {
                return leftMatrix.Subtract(right.AsMatrix());
            }

            return leftMatrix.Subtract(right.AsTensor());
        }

        if (left.IsTensor())
        {
            var leftTensor = left.AsTensor();

            if (right.IsScalar())
            {
                return leftTensor.Subtract(right.AsScalar());
            }

            if (right.IsVector())
            {
                return leftTensor.Subtract(right.AsVector());
            }

            if (right.IsMatrix())
            {
                return leftTensor.Subtract(right.AsMatrix());
            }

            return leftTensor.Subtract(right.AsTensor());
        }

        throw new UnreachableException();
    }

    public static ITensor<TNumber> operator *(ITensor<TNumber> left, ITensor<TNumber> right)
    {
        if (left.IsScalar())
        {
            var leftScalar = left.AsScalar();

            if (right.IsScalar())
            {
                return leftScalar.Multiply(right.AsScalar());
            }

            if (right.IsVector())
            {
                return leftScalar.Multiply(right.AsVector());
            }

            if (right.IsMatrix())
            {
                return leftScalar.Multiply(right.AsMatrix());
            }

            return leftScalar.Multiply(right.AsTensor());
        }

        if (left.IsVector())
        {
            var leftVector = left.AsVector();

            if (right.IsScalar())
            {
                return leftVector.Multiply(right.AsScalar());
            }

            throw new InvalidShapeException($"Cannot multiply shape {left.Shape} by shape {right.Shape}'");
        }

        if (left.IsMatrix())
        {
            var leftMatrix = left.AsMatrix();

            if (right.IsScalar())
            {
                return leftMatrix.Multiply(right.AsScalar());
            }

            throw new InvalidShapeException($"Cannot multiply shape {left.Shape} by shape {right.Shape}");
        }

        if (left.IsTensor())
        {
            var leftTensor = left.AsTensor();

            if (right.IsScalar())
            {
                return leftTensor.Multiply(right.AsScalar());
            }

            throw new InvalidShapeException($"Cannot multiply shape {left.Shape} by shape {right.Shape}");
        }

        throw new UnreachableException();
    }

    public static ITensor<TNumber> operator /(ITensor<TNumber> left, ITensor<TNumber> right)
    {
        if (left.IsScalar())
        {
            var leftScalar = left.AsScalar();

            if (right.IsScalar())
            {
                return leftScalar.Divide(right.AsScalar());
            }

            if (right.IsVector())
            {
                return leftScalar.Divide(right.AsVector());
            }

            if (right.IsMatrix())
            {
                return leftScalar.Divide(right.AsMatrix());
            }

            return leftScalar.Divide(right.AsTensor());
        }

        if (left.IsVector())
        {
            var leftVector = left.AsVector();

            if (right.IsScalar())
            {
                return leftVector.Divide(right.AsScalar());
            }

            throw new InvalidShapeException($"Cannot divide shape {left.Shape} by shape {right.Shape}");
        }

        if (left.IsMatrix())
        {
            var leftMatrix = left.AsMatrix();

            if (right.IsScalar())
            {
                return leftMatrix.Divide(right.AsScalar());
            }

            throw new InvalidShapeException($"Cannot divide shape {left.Shape} by shape {right.Shape}");
        }

        if (left.IsTensor())
        {
            var leftTensor = left.AsTensor();

            if (right.IsScalar())
            {
                return leftTensor.Divide(right.AsScalar());
            }

            throw new InvalidShapeException($"Cannot divide shape {left.Shape} by shape {right.Shape}");
        }

        throw new UnreachableException();
    }

    public static ITensor<TNumber> operator -(ITensor<TNumber> tensor)
    {
        if (tensor.IsScalar())
        {
            return tensor.AsScalar().Negate();
        }

        if (tensor.IsVector())
        {
            return tensor.AsVector().Negate();
        }

        if (tensor.IsMatrix())
        {
            return tensor.AsMatrix().Negate();
        }

        if (tensor.IsTensor())
        {
            return tensor.AsTensor().Negate();
        }

        throw new UnreachableException();
    }

#pragma warning restore CS1591

    /// <summary>
    /// Gets the transpose of the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <returns>The transpose of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Transpose()
    {
        return new Tensor<TNumber>(this, new Shape(Shape.Dimensions.Reverse().ToArray()));
    }

    /// <summary>
    /// Copies the values of the <see cref="ITensor{TNumber}"/> to an <see cref="Array"/>.
    /// </summary>
    /// <returns>An array with the same data as the <see cref="ITensor{TNumber}"/>.</returns>
    /// <exception cref="InvalidOperationException">The operation could not be completed.</exception>
    public Array ToArray();

    /// <summary>
    /// Creates an instance of <see cref="Vector{TNumber}"/> from the <see cref="ITensor{TNumber}"/>,
    /// assuming that the <see cref="ITensor{TNumber}"/> is a scalar.
    /// </summary>
    /// <returns>The <see cref="ITensor{TNumber}"/> instance as a <see cref="Vector{TNumber}"/>.</returns>
    /// <exception cref="InvalidShapeException">Throws when the shape of the <see cref="ITensor{TNumber}"/> is invalid.</exception>
    public Scalar<TNumber> AsScalar()
    {
        if (!Shape.IsScalar)
        {
            throw new InvalidShapeException($"The tensor must be a scalar, but got shape {Shape}");
        }

        return new Scalar<TNumber>(Handle, Backend);
    }

    /// <summary>
    /// Creates an instance of <see cref="Vector{TNumber}"/> from the <see cref="ITensor{TNumber}"/>,
    /// assuming that the <see cref="ITensor{TNumber}"/> is 1-dimensional.
    /// </summary>
    /// <returns>The <see cref="ITensor{TNumber}"/> instance as a <see cref="Vector{TNumber}"/>.</returns>
    /// <exception cref="InvalidShapeException">Throws when the shape of the <see cref="ITensor{TNumber}"/> is invalid.</exception>
    public Vector<TNumber> AsVector()
    {
        if (!Shape.IsVector)
        {
            throw new InvalidShapeException(
                $"The tensor must be 1-dimensional to be converted to a vector, but got shape {Shape}");
        }

        return new Vector<TNumber>(Shape.Dimensions[0], Handle, Backend);
    }

    /// <summary>
    /// Creates an instance of <see cref="Matrix{TNumber}"/> from the <see cref="ITensor{TNumber}"/>,
    /// assuming that the <see cref="ITensor{TNumber}"/> is 2-dimensional.
    /// </summary>
    /// <returns>The <see cref="ITensor{TNumber}"/> instance as a <see cref="Matrix{TNumber}"/>.</returns>
    /// <exception cref="InvalidShapeException">Throws when the shape of the <see cref="ITensor{TNumber}"/> is invalid.</exception>
    public Matrix<TNumber> AsMatrix()
    {
        if (!Shape.IsMatrix)
        {
            throw new InvalidShapeException(
                $"The tensor must be 2-dimensional to be converted to a matrix, but got shape {Shape}");
        }

        return new Matrix<TNumber>(Shape.Dimensions[0], Shape.Dimensions[1], Handle, Backend);
    }

    /// <summary>
    /// Creates an instance of <see cref="Tensor{TNumber}"/> from the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <returns>The <see cref="ITensor{TNumber}"/> instance as a <see cref="Tensor{TNumber}"/>.</returns>
    public Tensor<TNumber> AsTensor()
    {
        return new Tensor<TNumber>(this, Shape);
    }

    /// <summary>
    /// Checks if the <see cref="ITensor{TNumber}"/> is a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <returns><c>true</c> if the <see cref="ITensor{TNumber}"/> is a <see cref="Scalar{TNumber}"/> else, <c>false</c>.</returns>
    public bool IsScalar()
    {
        return Shape.IsScalar;
    }

    /// <summary>
    /// Checks if the <see cref="ITensor{TNumber}"/> is a <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <returns><c>true</c> if the <see cref="ITensor{TNumber}"/> is a <see cref="Vector{TNumber}"/> else, <c>false</c>.</returns>
    public bool IsVector()
    {
        return Shape.IsVector;
    }

    /// <summary>
    /// Checks if the <see cref="ITensor{TNumber}"/> is a <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <returns><c>true</c> if the <see cref="ITensor{TNumber}"/> is a <see cref="Matrix{TNumber}"/> else, <c>false</c>.</returns>
    public bool IsMatrix()
    {
        return Shape.IsMatrix;
    }

    /// <summary>
    /// Checks if the <see cref="ITensor{TNumber}"/> is a <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <returns><c>true</c> if the <see cref="ITensor{TNumber}"/> is a <see cref="Tensor{TNumber}"/> else, <c>false</c>.</returns>
    public bool IsTensor()
    {
        return Shape.IsTensor;
    }
}