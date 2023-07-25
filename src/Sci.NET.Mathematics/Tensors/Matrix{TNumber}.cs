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
        Handle = Backend.Storage.Allocate<TNumber>(Shape);
        IsMemoryOwner = true;
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
        Handle = handle;
        IsMemoryOwner = true;
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
    public IMemoryBlock<TNumber> Handle { get; private set; }

    /// <inheritdoc />
    public ITensorBackend Backend { get; }

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

    /// <summary>
    /// Gets the debugger display object.
    /// </summary>
    [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
#pragma warning disable IDE0051, RCS1213
    private Array DebuggerDisplayObject => ToArray();
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

        if (newDevice.Name == Device.Name)
        {
            return;
        }

        var newBackend = newDevice.GetTensorBackend();
        var oldHandle = Handle;
        var newHandle = newBackend.Storage.Allocate<TNumber>(Shape);
        using var tempTensor = new Tensor<TNumber>(newHandle, Shape, newBackend);

        newHandle.CopyFromSystemMemory(Handle.ToSystemMemory());
        Handle = newHandle;
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
        if (disposing && IsMemoryOwner)
        {
            Handle.Dispose();
        }
    }
}