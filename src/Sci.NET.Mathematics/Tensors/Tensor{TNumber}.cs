// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Memory.ReferenceCounting;
using Sci.NET.Mathematics.BLAS.Layout;
using Sci.NET.Mathematics.Tensors.Backends;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// An Rank-N implementation of <see cref="ITensor{TNumber}"/>.
/// </summary>
/// <typeparam name="TNumber">The type of number stored by the tensor.</typeparam>
[PublicAPI]
public class Tensor<TNumber> : ITensor<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly Shape _shape;

    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor{TNumber}"/> class.
    /// </summary>
    /// <param name="shape">The shape of the <see cref="Tensor{TNumber}"/>.</param>
    public Tensor(Shape shape)
    {
        _shape = shape;
        Handle = TensorBackend.Instance.Create<TNumber>(shape);
        ReferenceCount = new ReferenceCount();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor{TNumber}"/> class.
    /// </summary>
    /// <param name="handle">The handle to the existing data.</param>
    /// <param name="shape">The <see cref="Shape"/> of the <see cref="Tensor{TNumber}"/>.</param>
    public Tensor(TypedMemoryHandle<TNumber> handle, Shape shape)
    {
        _shape = shape;
        Handle = handle;
        ReferenceCount = new ReferenceCount();
    }

    /// <summary>
    /// Finalizes an instance of the <see cref="Tensor{TNumber}"/> class.
    /// </summary>
    ~Tensor()
    {
        Dispose(false);
    }

    /// <inheritdoc />
    public ReferenceCount ReferenceCount { get; }

    /// <inheritdoc />
    public TypedMemoryHandle<TNumber> Handle { get; }

    /// <inheritdoc />
    public int[] Dimensions => _shape.Dimensions;

    /// <inheritdoc />
    public int Rank => _shape.Rank;

    /// <inheritdoc />
    public long ElementCount => _shape.ElementCount;

    /// <inheritdoc />
    public long[] Strides => _shape.Strides;

    /// <inheritdoc />
    public bool IsScalar => _shape.IsScalar;

    /// <inheritdoc />
    public bool IsVector => _shape.IsVector;

    /// <inheritdoc />
    public bool IsMatrix => _shape.IsMatrix;

    /// <inheritdoc/>
    public TransposeType TransposeType => TransposeType.None;

    /// <summary>
    /// Gets the shape of the tensor.
    /// </summary>
    /// <returns>The shape of the tensor.</returns>
#pragma warning disable CA1024
    public Shape GetShape()
#pragma warning restore CA1024
    {
        return _shape;
    }

    /// <inheritdoc cref="Shape.GetIndices"/>
    public int[] GetIndices(long linearIndex)
    {
        return _shape.GetIndices(linearIndex);
    }

    /// <inheritdoc cref="Shape.GetLinearIndex"/>
    public long GetLinearIndex(params int[] indices)
    {
        return _shape.GetLinearIndex(indices);
    }

    /// <summary>
    /// Determines if the shape of the current tensor is equal to the <paramref name="other"/> shape.
    /// </summary>
    /// <param name="other">The shape to compare the shape of this tensor to.</param>
    /// <returns>
    /// <c>true</c> if the shape of this instance is equal to the
    /// <paramref name="other"/> shape, else <c>false</c>.
    /// </returns>
    public bool ShapeEquals(Shape other)
    {
        return _shape.Dimensions.SequenceEqual(other);
    }

    /// <summary>
    /// Determines if the shape of the current tensor is equal to the shape of the <paramref name="other"/> tensor.
    /// </summary>
    /// <param name="other">The tensor to compare the shape to.</param>
    /// <returns>
    /// <c>true</c> if the shape of this instance is equal to the
    /// shape of the <paramref name="other"/> tensor, else <c>false</c>.
    /// </returns>
    public bool ShapeEquals(Tensor<TNumber> other)
    {
        return ShapeEquals(other._shape);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases the unmanaged resources used by the <see cref="Tensor{TNumber}"/> and optionally releases the managed resources.
    /// </summary>
    /// <param name="disposing">Determines if the method is disposing.</param>
    protected virtual void Dispose(bool disposing)
    {
        ReleaseUnmanagedResources();
        if (disposing)
        {
            // Dispose managed resources.
        }
    }

    private void ReleaseUnmanagedResources()
    {
        ReferenceCount.Decrement();

        if (ReferenceCount.IsZero())
        {
            TensorBackend.Instance.Free(Handle);
        }
    }
}