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
    /// <summary>
    /// Initializes a new instance of the <see cref="Scalar{TNumber}"/> class.
    /// </summary>
    /// <param name="backend">The backend type to use for the <see cref="Vector{TNumber}"/>.</param>
    public Scalar(ITensorBackend? backend = null)
    {
        Shape = Shape.Scalar;
        Backend = backend ?? Tensor.DefaultBackend;
        Handle = Backend.Storage.Allocate<TNumber>(Shape);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Scalar{TNumber}"/> class.
    /// </summary>
    /// <param name="value">The value of the <see cref="Vector{TNumber}"/>.</param>
    /// <param name="backend">The backend type to use for the <see cref="Vector{TNumber}"/>.</param>
    public Scalar(TNumber value, ITensorBackend? backend = null)
    {
        Shape = Shape.Scalar;
        Backend = backend ?? Tensor.DefaultBackend;
        Handle = Backend.Storage.Allocate<TNumber>(Shape);
        IsMemoryOwner = true;
        Handle[0] = value;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Scalar{TNumber}"/> class.
    /// </summary>
    /// <param name="handle">The memory handle to use for the <see cref="Vector{TNumber}"/>.</param>
    /// <param name="backend">The backend type to use for the <see cref="Vector{TNumber}"/>.</param>
    public Scalar(IMemoryBlock<TNumber> handle, ITensorBackend backend)
    {
        Shape = Shape.Scalar;
        Backend = backend;
        Handle = handle;
        IsMemoryOwner = true;
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
    public IMemoryBlock<TNumber> Handle { get; private set; }

    /// <inheritdoc />
    public ITensorBackend Backend { get; }

    /// <inheritdoc />
    public bool IsMemoryOwner { get; private set; }

    /// <summary>
    /// Gets the length of the <see cref="Vector{TNumber}"/>.
    /// </summary>
    public int Length => Shape[0];

    /// <summary>
    /// Gets the value of the <see cref="Scalar{TNumber}"/>.
    /// </summary>
    public TNumber Value => Handle[0];

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
    /// <param name="index">The index of the scalar to get.</param>
    public Scalar<TNumber> this[int index] => Tensor.Slice(this, index).ToScalar();
#pragma warning restore CA2000, CA1043

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