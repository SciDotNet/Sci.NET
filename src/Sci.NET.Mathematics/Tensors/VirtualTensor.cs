// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Memory.ReferenceCounting;
using Sci.NET.Mathematics.Tensors.Backends;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// Implements a virtual tensor which uses the <see cref="IMemoryBlock{T}"/>
/// of a concrete tensor as its storage. This is useful for reshaping or
/// permuting a tensor.
/// </summary>
/// <typeparam name="TNumber">The type of number stored by the tensor.</typeparam>
[PublicAPI]
public class VirtualTensor<TNumber> : ITensor<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly Shape _shape;

    /// <summary>
    /// Initializes a new instance of the <see cref="VirtualTensor{TNumber}"/> class.
    /// </summary>
    /// <param name="tensor">The tensor to create a reference to.</param>
    /// <param name="shape">The shape of the tensor.</param>
    public VirtualTensor(ITensor<TNumber> tensor, Shape shape)
    {
        if (shape.ElementCount != tensor.ElementCount)
        {
            throw new ArgumentException(
                "The shape and storage must have the same number of elements.",
                nameof(tensor));
        }

        _shape = shape;
        ReferenceCount = tensor.ReferenceCount;
        ReferenceCount.Increment();
        Data = tensor.Data;
    }

    /// <inheritdoc />
    public ReferenceCount ReferenceCount { get; }

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

    /// <inheritdoc />
    public IMemoryBlock<TNumber> Data { get; }

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
            TensorBackend.Instance.Free(Data);
        }
    }
}