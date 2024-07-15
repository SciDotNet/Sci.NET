// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// Represents a matrix.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
[PublicAPI]
[DebuggerDisplay("{ToArray()}")]
public sealed class Matrix<TNumber> : ITensor<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly Guid _id = Guid.NewGuid();

    /// <summary>
    /// Initializes a new instance of the <see cref="Matrix{TNumber}"/> class.
    /// </summary>
    /// <param name="rows">The number of rows in the <see cref="Matrix{TNumber}"/>.</param>
    /// <param name="columns">The number of columns in the <see cref="Matrix{TNumber}"/>.</param>
    /// <param name="backend">The backend type to use for the <see cref="Matrix{TNumber}"/>.</param>
    public Matrix(int rows, int columns, ITensorBackend? backend = null)
    {
        Shape = new Shape(rows, columns);
        Backend = backend ?? Tensor.DefaultBackend;
        Memory = Backend.Storage.Allocate<TNumber>(Shape);
        IsMemoryOwner = true;
        Memory.Rent(_id);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Matrix{TNumber}"/> class.
    /// </summary>
    /// <param name="rows">The number of rows in the <see cref="Matrix{TNumber}"/>.</param>
    /// <param name="columns">The number of columns in the <see cref="Matrix{TNumber}"/>.</param>
    /// <param name="handle">The memory handle to use for the <see cref="Matrix{TNumber}"/>.</param>
    /// <param name="backend">The backend type to use for the <see cref="Matrix{TNumber}"/>.</param>
    public Matrix(int rows, int columns, IMemoryBlock<TNumber> handle, ITensorBackend backend)
    {
        Shape = new Shape(rows, columns);
        Backend = backend;
        Memory = handle;
        IsMemoryOwner = false;
        Memory.Rent(_id);
    }

    /// <summary>
    /// Finalizes an instance of the <see cref="Matrix{TNumber}"/> class.
    /// </summary>
    ~Matrix()
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
    public bool IsMemoryOwner { get; set; }

    /// <summary>
    /// Gets the number of rows in the <see cref="Matrix{TNumber}"/>.
    /// </summary>
    public int Rows => Shape[0];

    /// <summary>
    /// Gets the number of columns in the <see cref="Matrix{TNumber}"/>.
    /// </summary>
    public int Columns => Shape[1];

#pragma warning disable IDE0051, RCS1213
    [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
    private Array DebuggerDisplayObject => Shape.ElementCount < 10000 ? ToArray() : new[] { "The matrix too big to view" };
#pragma warning restore RCS1213, IDE0051

    /// <inheritdoc />
#pragma warning disable CA1043, CA2000
    public ITensor<TNumber> this[params int[] indices] => Tensor.Slice(this, indices);

    /// <summary>
    /// Gets the <see cref="Scalar{TNumber}"/> at the specified index.
    /// </summary>
    /// <param name="x">The row index.</param>
    /// <param name="y">The column index.</param>
    public Scalar<TNumber> this[int x, int y] => Tensor.Slice(this, x, y).ToScalar();

    /// <summary>
    /// Gets the <see cref="Vector{TNumber}"/> at the specified index.
    /// </summary>
    /// <param name="x">The row index.</param>
    public Vector<TNumber> this[int x] => Tensor.Slice(this, x).ToVector();

#pragma warning restore CA2000, CA1043

    /// <summary>
    /// Adds the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the addition.</returns>
    public static Matrix<TNumber> operator +(Matrix<TNumber> left, TNumber right)
    {
        using var rightScalar = new Scalar<TNumber>(right);
        rightScalar.To(left.Device);

        return left.Add(rightScalar);
    }

    /// <summary>
    /// Adds the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the addition.</returns>
    public static Matrix<TNumber> operator +(Matrix<TNumber> left, Scalar<TNumber> right)
    {
        return left.Add(right);
    }

    /// <summary>
    /// Adds the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the addition.</returns>
    public static Matrix<TNumber> operator +(Matrix<TNumber> left, Vector<TNumber> right)
    {
        return left.Add(right);
    }

    /// <summary>
    /// Adds the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the addition.</returns>
    public static Matrix<TNumber> operator +(Matrix<TNumber> left, Matrix<TNumber> right)
    {
        return left.Add(right);
    }

    /// <summary>
    /// Adds the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the addition.</returns>
    public static Tensor<TNumber> operator +(Matrix<TNumber> left, Tensor<TNumber> right)
    {
        return left.Add(right);
    }

    /// <summary>
    /// Subtracts the left operands from the right.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the subtraction.</returns>
    public static Matrix<TNumber> operator -(Matrix<TNumber> left, TNumber right)
    {
        using var rightScalar = new Scalar<TNumber>(right);
        rightScalar.To(left.Device);

        return left.Subtract(rightScalar);
    }

    /// <summary>
    /// Subtracts the left operands from the right.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the subtraction.</returns>
    public static Matrix<TNumber> operator -(Matrix<TNumber> left, Scalar<TNumber> right)
    {
        return left.Subtract(right);
    }

    /// <summary>
    /// Subtracts the left operands from the right.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the subtraction.</returns>
    public static Matrix<TNumber> operator -(Matrix<TNumber> left, Vector<TNumber> right)
    {
        return left.Subtract(right);
    }

    /// <summary>
    /// Subtracts the left operands from the right.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the subtraction.</returns>
    public static Matrix<TNumber> operator -(Matrix<TNumber> left, Matrix<TNumber> right)
    {
        return left.Subtract(right);
    }

    /// <summary>
    /// Subtracts the left operands from the right.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the subtraction.</returns>
    public static Tensor<TNumber> operator -(Matrix<TNumber> left, Tensor<TNumber> right)
    {
        return left.Subtract(right);
    }

    /// <summary>
    /// Multiplies the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the multiplication.</returns>
    public static Matrix<TNumber> operator *(Matrix<TNumber> left, TNumber right)
    {
        using var rightScalar = new Scalar<TNumber>(right);
        rightScalar.To(left.Device);

        return left.Multiply(rightScalar);
    }

    /// <summary>
    /// Multiplies the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the multiplication.</returns>
    public static Matrix<TNumber> operator *(Matrix<TNumber> left, Scalar<TNumber> right)
    {
        return left.Multiply(right);
    }

    /// <summary>
    /// Multiplies the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the multiplication.</returns>
    public static Matrix<TNumber> operator *(Matrix<TNumber> left, Vector<TNumber> right)
    {
        return left.Multiply(right);
    }

    /// <summary>
    /// Multiplies the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the multiplication.</returns>
    public static Matrix<TNumber> operator *(Matrix<TNumber> left, Matrix<TNumber> right)
    {
        return left.Multiply(right);
    }

    /// <summary>
    /// Multiplies the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the multiplication.</returns>
    public static Tensor<TNumber> operator *(Matrix<TNumber> left, Tensor<TNumber> right)
    {
        return left.Multiply(right);
    }

    /// <summary>
    /// Divides the left operand by the right operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the division.</returns>
    public static Matrix<TNumber> operator /(Matrix<TNumber> left, TNumber right)
    {
        using var rightScalar = new Scalar<TNumber>(right);
        rightScalar.To(left.Device);

        return left.Divide(rightScalar);
    }

    /// <summary>
    /// Divides the left operand by the right operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the division.</returns>
    public static Matrix<TNumber> operator /(Matrix<TNumber> left, Scalar<TNumber> right)
    {
        return left.Divide(right);
    }

    /// <summary>
    /// Divides the left operand by the right operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the division.</returns>
    public static Matrix<TNumber> operator /(Matrix<TNumber> left, Vector<TNumber> right)
    {
        return left.Divide(right);
    }

    /// <summary>
    /// Divides the left operand by the right operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the division.</returns>
    public static Matrix<TNumber> operator /(Matrix<TNumber> left, Matrix<TNumber> right)
    {
        return left.Divide(right);
    }

    /// <summary>
    /// Divides the left operand by the right operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the division.</returns>
    public static Tensor<TNumber> operator /(Matrix<TNumber> left, Tensor<TNumber> right)
    {
        return left.Divide(right);
    }

    /// <inheritdoc />
    public Array ToArray()
    {
        return Tensor.ToArray(this);
    }

    /// <inheritdoc/>
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

    /// <inheritdoc />
    public override string ToString()
    {
        return Tensor.ToString(this);
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