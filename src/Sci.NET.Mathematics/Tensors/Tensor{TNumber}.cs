// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// A rank-N tensor, which is an immutable N-Dimensional array.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
[PublicAPI]
public class Tensor<TNumber> : ITensor<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly WeakReference<IMemoryBlock<TNumber>> _weakHandle;

    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor{TNumber}"/> class.
    /// </summary>
    /// <param name="backend">The backend type to use for the <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="shape">The dimensions of the <see cref="Tensor{TNumber}"/>.</param>
    public Tensor(ITensorBackend? backend = null, params int[] shape)
    {
        Shape = new Shape(shape);
        Backend = backend ?? Tensor.DefaultBackend;
        Handle = Backend.Storage.Allocate<TNumber>(Shape);
        _weakHandle = new WeakReference<IMemoryBlock<TNumber>>(Handle);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor{TNumber}"/> class.
    /// </summary>
    /// <param name="shape">The shape of the <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="backend">The backend type to use for the <see cref="ITensor{TNumber}"/>.</param>
    public Tensor(Shape shape, ITensorBackend? backend = null)
        : this(backend, shape.Dimensions)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor{TNumber}"/> class.
    /// </summary>
    /// <param name="previousTensor">The previous tensor to copy.</param>
    /// <param name="newShape">The new shape of the <see cref="Tensor{TNumber}"/>.</param>
    public Tensor(ITensor<TNumber> previousTensor, Shape newShape)
    {
        if (newShape.ElementCount != previousTensor.Shape.ElementCount)
        {
            throw new ArgumentException("The new shape must have the same number of elements as the previous tensor.");
        }

        Handle = previousTensor.Handle;
        Backend = previousTensor.Backend;
        Shape = newShape;
        _weakHandle = new WeakReference<IMemoryBlock<TNumber>>(Handle);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor{TNumber}"/> class.
    /// </summary>
    /// <param name="memoryBlock">The memory block for the <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="shape">The shape of the <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="backend">The <see cref="ITensorBackend"/> instance which the <see cref="ITensor{TNumber}"/> uses.</param>
    public Tensor(IMemoryBlock<TNumber> memoryBlock, Shape shape, ITensorBackend backend)
    {
        Handle = memoryBlock;
        Shape = shape;
        Backend = backend;
        _weakHandle = new WeakReference<IMemoryBlock<TNumber>>(Handle);
    }

    /// <inheritdoc />
    public Shape Shape { get; }

    /// <inheritdoc />
    public IMemoryBlock<TNumber> Handle { get; }

    /// <inheritdoc />
    public ITensorBackend Backend { get; }

    /// <inheritdoc/>
    public IDevice Device => Backend.Device;

    /// <inheritdoc />
#pragma warning disable CA1043
    public ITensor<TNumber> this[params int[] indices] => Tensor.Slice(this, indices);

#pragma warning restore CA1043

#pragma warning disable CS1591
    public static Tensor<TNumber> operator +(Tensor<TNumber> left, Scalar<TNumber> right)
    {
        return left.Add(right);
    }

    public static Tensor<TNumber> operator +(Scalar<TNumber> left, Tensor<TNumber> right)
    {
        return left.Add(right);
    }

    public static Tensor<TNumber> operator +(Tensor<TNumber> left, Tensor<TNumber> right)
    {
        return left.Add(right);
    }

    public static Tensor<TNumber> operator -(Tensor<TNumber> left, Scalar<TNumber> right)
    {
        return left.Subtract(right);
    }

    public static Tensor<TNumber> operator -(Scalar<TNumber> left, Tensor<TNumber> right)
    {
        return left.Subtract(right);
    }

    public static Tensor<TNumber> operator -(Tensor<TNumber> left, Tensor<TNumber> right)
    {
        return left.Subtract(right);
    }

    public static Tensor<TNumber> operator *(Tensor<TNumber> left, Scalar<TNumber> right)
    {
        return right.Multiply(left);
    }

    public static Tensor<TNumber> operator *(Scalar<TNumber> left, Tensor<TNumber> right)
    {
        return right.Multiply(left);
    }

#pragma warning restore CS1591

    /// <inheritdoc />
    public unsafe Array ToArray()
    {
        if (Shape.IsScalar)
        {
            throw new InvalidOperationException("Cannot convert a scalar to an array.");
        }

        var result = Array.CreateInstance(typeof(TNumber), Shape.Dimensions);
        result.Initialize();

        var startIndex = Shape.DataOffset;
        var endIndex = startIndex + Shape.ElementCount;
        var bytesToCopy = Unsafe.SizeOf<TNumber>() * (endIndex - startIndex);
        var systemMemoryClone = Handle.ToSystemMemory();

        var sourcePointer = Unsafe.AsPointer(
            ref Unsafe.Add(ref Unsafe.AsRef<TNumber>(systemMemoryClone.ToPointer()), (nuint)startIndex));
        var destinationPointer = Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(result));

        Buffer.MemoryCopy(
            sourcePointer,
            destinationPointer,
            Unsafe.SizeOf<TNumber>() * result.LongLength,
            bytesToCopy);

        return result;
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
    /// <param name="disposing">A value indicating whether the instance is disposing.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            Handle.Dispose();
            _ = _weakHandle.TryGetTarget(out _);
        }
    }
}