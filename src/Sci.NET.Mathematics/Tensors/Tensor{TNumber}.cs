// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// A rank-N tensor, which is an immutable N-Dimensional array.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
[PublicAPI]
public sealed class Tensor<TNumber> : ITensor<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly Guid _id = Guid.NewGuid();

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
        IsMemoryOwner = true;
        Handle.Rent(_id);
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
        IsMemoryOwner = false;
        Handle.Rent(_id);
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
        IsMemoryOwner = false;
        Handle.Rent(_id);
    }

    /// <summary>
    /// Finalizes an instance of the <see cref="Tensor{TNumber}"/> class.
    /// </summary>
    ~Tensor()
    {
        Dispose(false);
    }

    /// <inheritdoc />
    public Shape Shape { get; }

    /// <inheritdoc />
    public IMemoryBlock<TNumber> Handle { get; private set; }

    /// <inheritdoc />
    public ITensorBackend Backend { get; private set; }

    /// <inheritdoc />
    public bool IsMemoryOwner { get; private set; }

    /// <inheritdoc/>
    public IDevice Device => Backend.Device;

#pragma warning disable IDE0051, RCS1213
    [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
    private Array DebuggerDisplayObject => ToArray();
#pragma warning restore RCS1213, IDE0051

    /// <inheritdoc />
#pragma warning disable CA1043
    public ITensor<TNumber> this[params int[] indices] => Tensor.Slice(this, indices);
#pragma warning restore CA1043

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
    /// Releases the unmanaged resources used by the <see cref="Tensor{TNumber}"/> and optionally releases the managed resources.
    /// </summary>
    /// <param name="disposing">A value indicating whether the instance is disposing.</param>
    private void Dispose(bool disposing)
    {
        Handle.Release(_id);

        if (disposing && IsMemoryOwner)
        {
            Handle.Dispose();
        }
    }
}