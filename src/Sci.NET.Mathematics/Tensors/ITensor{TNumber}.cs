// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Backends;
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
    /// Gets a value indicating whether the <see cref="ITensor{TNumber}"/> owns the memory it points to.
    /// </summary>
    public bool IsMemoryOwner { get; }

    /// <summary>
    /// Gets the slice of the <see cref="ITensor{TNumber}"/> at the specified indices.
    /// </summary>
    /// <param name="indices">The indices of the <see cref="ITensor{TNumber}"/> to slice.</param>
#pragma warning disable CA1043
    public ITensor<TNumber> this[params int[] indices] { get; }
#pragma warning restore CA1043

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
    public Scalar<TNumber> ToScalar()
    {
        if (!Shape.IsScalar)
        {
            throw new InvalidShapeException($"The tensor must be a scalar, but got shape {Shape}");
        }

        DetachMemory();

        return new Scalar<TNumber>(Handle, Backend);
    }

    /// <summary>
    /// Creates an instance of <see cref="Vector{TNumber}"/> from the <see cref="ITensor{TNumber}"/>,
    /// assuming that the <see cref="ITensor{TNumber}"/> is 1-dimensional.
    /// </summary>
    /// <returns>The <see cref="ITensor{TNumber}"/> instance as a <see cref="Vector{TNumber}"/>.</returns>
    /// <exception cref="InvalidShapeException">Throws when the shape of the <see cref="ITensor{TNumber}"/> is invalid.</exception>
    public Vector<TNumber> ToVector()
    {
        if (!Shape.IsVector)
        {
            throw new InvalidShapeException(
                $"The tensor must be 1-dimensional to be converted to a vector, but got shape {Shape}");
        }

        DetachMemory();

        return new Vector<TNumber>(Shape.Dimensions[0], Handle, Backend);
    }

    /// <summary>
    /// Creates an instance of <see cref="Matrix{TNumber}"/> from the <see cref="ITensor{TNumber}"/>,
    /// assuming that the <see cref="ITensor{TNumber}"/> is 2-dimensional.
    /// </summary>
    /// <returns>The <see cref="ITensor{TNumber}"/> instance as a <see cref="Matrix{TNumber}"/>.</returns>
    /// <exception cref="InvalidShapeException">Throws when the shape of the <see cref="ITensor{TNumber}"/> is invalid.</exception>
    public Matrix<TNumber> ToMatrix()
    {
        if (!Shape.IsMatrix)
        {
            throw new InvalidShapeException(
                $"The tensor must be 2-dimensional to be converted to a matrix, but got shape {Shape}");
        }

        DetachMemory();

        return new Matrix<TNumber>(Shape.Dimensions[0], Shape.Dimensions[1], Handle, Backend);
    }

    /// <summary>
    /// Creates an instance of <see cref="Tensor{TNumber}"/> from the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <returns>The <see cref="ITensor{TNumber}"/> instance as a <see cref="Tensor{TNumber}"/>.</returns>
    public Tensor<TNumber> ToTensor()
    {
        DetachMemory();

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

    /// <summary>
    /// Detaches the memory from the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    protected void DetachMemory();
}