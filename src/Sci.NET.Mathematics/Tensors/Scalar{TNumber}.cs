// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// Represents a scalar.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
[PublicAPI]
public sealed class Scalar<TNumber> : ITensor<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly Guid _id = Guid.NewGuid();

    /// <summary>
    /// Initializes a new instance of the <see cref="Scalar{TNumber}"/> class.
    /// </summary>
    /// <param name="backend">The backend type to use for the <see cref="Vector{TNumber}"/>.</param>
    public Scalar(ITensorBackend? backend = null)
    {
        Shape = Shape.Scalar();
        Backend = backend ?? Tensor.DefaultBackend;
        Memory = Backend.Storage.Allocate<TNumber>(Shape);
        IsMemoryOwner = true;
        Memory.Rent(_id);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Scalar{TNumber}"/> class.
    /// </summary>
    /// <param name="value">The value of the <see cref="Vector{TNumber}"/>.</param>
    /// <param name="backend">The backend type to use for the <see cref="Vector{TNumber}"/>.</param>
    public Scalar(TNumber value, ITensorBackend? backend = null)
    {
        Shape = Shape.Scalar();
        Backend = backend ?? Tensor.DefaultBackend;
        Memory = Backend.Storage.Allocate<TNumber>(Shape);
        IsMemoryOwner = true;
        Memory.Rent(_id);

        using var systemMemory = new SystemMemoryBlock<TNumber>(1);
        systemMemory[0] = value;

        Memory.CopyFromSystemMemory(systemMemory);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Scalar{TNumber}"/> class.
    /// </summary>
    /// <param name="handle">The memory handle to use for the <see cref="Vector{TNumber}"/>.</param>
    /// <param name="backend">The backend type to use for the <see cref="Vector{TNumber}"/>.</param>
    public Scalar(IMemoryBlock<TNumber> handle, ITensorBackend backend)
    {
        Shape = Shape.Scalar();
        Backend = backend;
        Memory = handle;
        IsMemoryOwner = false;
        Memory.Rent(_id);
    }

    /// <summary>
    /// Finalizes an instance of the <see cref="Scalar{TNumber}"/> class.
    /// </summary>
    ~Scalar()
    {
        Dispose(false);
    }

    /// <inheritdoc />
    public IDevice Device => Backend.Device;

    /// <inheritdoc />
    public Shape Shape { get; }

    /// <inheritdoc />
    public IMemoryBlock<TNumber> Memory { get; private set; }

    /// <inheritdoc />
    public ITensorBackend Backend { get; private set; }

    /// <inheritdoc />
    public bool IsMemoryOwner { get; private set; }

    /// <summary>
    /// Gets the length of the <see cref="Vector{TNumber}"/>.
    /// </summary>
    public int Length => Shape[0];

    /// <summary>
    /// Gets the value of the <see cref="Scalar{TNumber}"/>.
    /// </summary>
    public TNumber Value => Memory.ToSystemMemory()[0];

#pragma warning disable IDE0051, RCS1213
    [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
    private Array DebuggerDisplayObject => ToArray();
#pragma warning restore RCS1213, IDE0051

    /// <inheritdoc />
#pragma warning disable CA1043, CA2000
    public ITensor<TNumber> this[params int[] indices] => Tensor.Slice(this, indices);

    /// <summary>
    /// Gets the <see cref="Scalar{TNumber}"/> at the specified index.
    /// </summary>
    /// <param name="index">The index of the scalar to get.</param>
    public Scalar<TNumber> this[int index] => Tensor.Slice(this, index).ToScalar();
#pragma warning restore CA2000, CA1043

    public static Scalar<TNumber> operator +(Scalar<TNumber> left, Scalar<TNumber> right)
    {
        return left.Add(right);
    }

    public static Scalar<TNumber> operator +(Scalar<TNumber> left, TNumber right)
    {
        using var rightScalar = new Scalar<TNumber>(right);
        rightScalar.To(left.Device);

        return left.Add(rightScalar);
    }

    public static Vector<TNumber> operator +(Scalar<TNumber> left, Vector<TNumber> right)
    {
        return left.Add(right);
    }

    public static Matrix<TNumber> operator +(Scalar<TNumber> left, Matrix<TNumber> right)
    {
        return right.Add(left);
    }

    public static Tensor<TNumber> operator +(Scalar<TNumber> left, Tensor<TNumber> right)
    {
        return right.Add(left);
    }

    public static Scalar<TNumber> operator -(Scalar<TNumber> left, TNumber right)
    {
        using var rightScalar = new Scalar<TNumber>(right);
        rightScalar.To(left.Device);

        return left.Subtract(rightScalar);
    }

    public static Scalar<TNumber> operator -(Scalar<TNumber> left, Scalar<TNumber> right)
    {
        return left.Subtract(right);
    }

    public static Vector<TNumber> operator -(Scalar<TNumber> left, Vector<TNumber> right)
    {
        return right.Subtract(left);
    }

    public static Matrix<TNumber> operator -(Scalar<TNumber> left, Matrix<TNumber> right)
    {
        return right.Subtract(left);
    }

    public static Tensor<TNumber> operator -(Scalar<TNumber> left, Tensor<TNumber> right)
    {
        return right.Subtract(left);
    }

    public static Scalar<TNumber> operator *(Scalar<TNumber> left, TNumber right)
    {
        using var rightScalar = new Scalar<TNumber>(right);
        rightScalar.To(left.Device);

        return left.Multiply(rightScalar);
    }

    public static Scalar<TNumber> operator *(Scalar<TNumber> left, Scalar<TNumber> right)
    {
        return left.Multiply(right);
    }

    public static Vector<TNumber> operator *(Scalar<TNumber> left, Vector<TNumber> right)
    {
        return left.Multiply(right);
    }

    public static Matrix<TNumber> operator *(Scalar<TNumber> left, Matrix<TNumber> right)
    {
        return right.Multiply(left);
    }

    public static Tensor<TNumber> operator *(Scalar<TNumber> left, Tensor<TNumber> right)
    {
        return right.Multiply(left);
    }

    public static Scalar<TNumber> operator /(Scalar<TNumber> left, TNumber right)
    {
        using var rightScalar = new Scalar<TNumber>(right);
        rightScalar.To(left.Device);

        return left.Divide(rightScalar);
    }

    public static Scalar<TNumber> operator /(Scalar<TNumber> left, Scalar<TNumber> right)
    {
        return left.Divide(right);
    }

    public static Vector<TNumber> operator /(Scalar<TNumber> left, Vector<TNumber> right)
    {
        return right.Divide(left);
    }

    public static Matrix<TNumber> operator /(Scalar<TNumber> left, Matrix<TNumber> right)
    {
        return right.Divide(left);
    }

    public static Tensor<TNumber> operator /(Scalar<TNumber> left, Tensor<TNumber> right)
    {
        return right.Divide(left);
    }

    /// <inheritdoc />
    public Array ToArray()
    {
        return Tensor.ToArray(this);
    }

    /// <inheritdoc />
    void ITensor<TNumber>.DetachMemory()
    {
        IsMemoryOwner = false;
    }

    /// <inheritdoc />
    public void To<TDevice>()
        where TDevice : IDevice, new()
    {
        var newDevice = new TDevice();
        To(newDevice);
    }

    /// <inheritdoc />
    public void To(IDevice device)
    {
        if (device.Name == Device.Name)
        {
            return;
        }

        var newBackend = device.GetTensorBackend();
        var oldHandle = Memory;
        var newHandle = newBackend.Storage.Allocate<TNumber>(Shape);
        using var tempTensor = new Tensor<TNumber>(newHandle, Shape, newBackend);

        newHandle.CopyFromSystemMemory(Memory.ToSystemMemory());
        Memory = newHandle;
        Backend = newBackend;
        oldHandle.Dispose();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes of the <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="disposing">A value indicating whether the <see cref="Vector{TNumber}"/> is disposing.</param>
    private void Dispose(bool disposing)
    {
        Memory.Release(_id);

        if (disposing && IsMemoryOwner)
        {
            Memory.Dispose();
        }
    }
}