// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
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
    /// <param name="requiresGradient">A value indicating whether the <see cref="ITensor{TNumber}"/> requires a gradient.</param>
    /// <param name="shape">The dimensions of the <see cref="Tensor{TNumber}"/>.</param>
    public Tensor(ITensorBackend? backend = null, bool requiresGradient = false, params int[] shape)
    {
        Shape = new Shape(shape);
        Backend = backend ?? Tensor.DefaultBackend;
        Memory = Backend.Storage.Allocate<TNumber>(Shape);
        IsMemoryOwner = true;
        Memory.Rent(_id);
        RequiresGradient = requiresGradient;
        Gradient = RequiresGradient ? new Tensor<TNumber>(Shape, Backend, false) { IsGradient = true } : null;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor{TNumber}"/> class.
    /// </summary>
    /// <param name="shape">The shape of the <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="backend">The backend type to use for the <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="requiresGradient">A value indicating whether the <see cref="ITensor{TNumber}"/> requires a gradient.</param>
    public Tensor(Shape shape, ITensorBackend? backend = null, bool requiresGradient = false)
        : this(backend, requiresGradient, shape.Dimensions)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor{TNumber}"/> class.
    /// </summary>
    /// <param name="previousTensor">The previous tensor to copy.</param>
    /// <param name="newShape">The new shape of the <see cref="Tensor{TNumber}"/>.</param>
    /// <param name="overrideRequiresGradient">A value indicating whether the <see cref="Tensor{TNumber}"/> requires a gradient, this overrides the <paramref name="previousTensor"/> value.</param>
    public Tensor(ITensor<TNumber> previousTensor, Shape newShape, bool? overrideRequiresGradient = null)
    {
        if (newShape.ElementCount != previousTensor.Shape.ElementCount)
        {
            throw new ArgumentException("The new shape must have the same number of elements as the previous tensor.");
        }

        Memory = previousTensor.Memory;
        Backend = previousTensor.Backend;
        Shape = newShape;
        IsMemoryOwner = false;
        Memory.Rent(_id);
        RequiresGradient = previousTensor.RequiresGradient;
        Gradient = overrideRequiresGradient ?? RequiresGradient ? new Tensor<TNumber>(Shape, Backend, requiresGradient: false) { IsGradient = true } : null;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor{TNumber}"/> class.
    /// </summary>
    /// <param name="memoryBlock">The memory block for the <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="shape">The shape of the <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="backend">The <see cref="ITensorBackend"/> instance which the <see cref="ITensor{TNumber}"/> uses.</param>
    /// <param name="requiresGradient">A value indicating whether the <see cref="ITensor{TNumber}"/> requires a gradient.</param>
    public Tensor(IMemoryBlock<TNumber> memoryBlock, Shape shape, ITensorBackend backend, bool requiresGradient)
    {
        Memory = memoryBlock;
        Shape = shape;
        Backend = backend;
        IsMemoryOwner = false;
        Memory.Rent(_id);
        RequiresGradient = requiresGradient;
        Gradient = RequiresGradient ? new Tensor<TNumber>(Shape, Backend, false) { IsGradient = true } : null;
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
    public IMemoryBlock<TNumber> Memory { get; private set; }

    /// <inheritdoc />
    public ITensorBackend Backend { get; private set; }

    /// <inheritdoc />
    public bool IsMemoryOwner { get; private set; }

    /// <inheritdoc />
    [MemberNotNullWhen(true, nameof(RequiresGradient))]
    public ITensor<TNumber>? Gradient { get; internal set; }

    /// <inheritdoc />
    public bool RequiresGradient { get; }

    /// <inheritdoc />
    public bool IsGradient { get; init; }

    /// <inheritdoc/>
    ICollection<(string Name, ITensor<TNumber> Parent, Func<ITensor<TNumber>, ITensor<TNumber>> Gradient)> ITensor<TNumber>.Parents { get; } = new List<(string Name, ITensor<TNumber> Parent, Func<ITensor<TNumber>, ITensor<TNumber>> Gradient)>();

    /// <inheritdoc/>
    public IDevice Device => Backend.Device;

#pragma warning disable IDE0051, RCS1213
    [DebuggerBrowsable(DebuggerBrowsableState.Collapsed)]
    [ExcludeFromCodeCoverage]
    private Array Data => Shape.All(x => x < 10000) ? ToArray() : new[] { "The tensor too big to view" };
#pragma warning restore RCS1213, IDE0051

    /// <inheritdoc />
#pragma warning disable CA1043
    public ITensor<TNumber> this[params int[] indices] => Tensor.Slice(this, indices);
#pragma warning restore CA1043

    /// <summary>
    /// Adds the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the addition.</returns>
    public static Tensor<TNumber> operator +(Tensor<TNumber> left, Scalar<TNumber> right)
    {
        return left.Add(right);
    }

    /// <summary>
    /// Adds the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the addition.</returns>
    public static Tensor<TNumber> operator +(Tensor<TNumber> left, Vector<TNumber> right)
    {
        return left.Add(right);
    }

    /// <summary>
    /// Adds the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the addition.</returns>
    public static Tensor<TNumber> operator +(Tensor<TNumber> left, Matrix<TNumber> right)
    {
        return left.Add(right);
    }

    /// <summary>
    /// Adds the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the addition.</returns>
    public static Tensor<TNumber> operator +(Tensor<TNumber> left, Tensor<TNumber> right)
    {
        return left.Add(right);
    }

    /// <summary>
    /// Subtracts the left operand from the right operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the subtraction.</returns>
    public static Tensor<TNumber> operator -(Tensor<TNumber> left, Scalar<TNumber> right)
    {
        return left.Subtract(right);
    }

    /// <summary>
    /// Subtracts the left operand from the right operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the subtraction.</returns>
    public static Tensor<TNumber> operator -(Tensor<TNumber> left, Vector<TNumber> right)
    {
        return left.Subtract(right);
    }

    /// <summary>
    /// Subtracts the left operand from the right operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the subtraction.</returns>
    public static Tensor<TNumber> operator -(Tensor<TNumber> left, Matrix<TNumber> right)
    {
        return left.Subtract(right);
    }

    /// <summary>
    /// Subtracts the left operand from the right operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the subtraction.</returns>
    public static Tensor<TNumber> operator -(Tensor<TNumber> left, Tensor<TNumber> right)
    {
        return left.Subtract(right);
    }

    /// <summary>
    /// Multiplies the left operand by the right operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the multiplication.</returns>
    public static Tensor<TNumber> operator *(Tensor<TNumber> left, Scalar<TNumber> right)
    {
        return left.Multiply(right);
    }

    /// <summary>
    /// Multiplies the left operand by the right operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the multiplication.</returns>
    public static Tensor<TNumber> operator *(Tensor<TNumber> left, Vector<TNumber> right)
    {
        return left.Multiply(right);
    }

    /// <summary>
    /// Multiplies the left operand by the right operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the multiplication.</returns>
    public static Tensor<TNumber> operator *(Tensor<TNumber> left, Matrix<TNumber> right)
    {
        return left.Multiply(right);
    }

    /// <summary>
    /// Multiplies the left operand by the right operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the multiplication.</returns>
    public static Tensor<TNumber> operator *(Tensor<TNumber> left, Tensor<TNumber> right)
    {
        return left.Multiply(right);
    }

    /// <summary>
    /// Divides the left operand by the right operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the division.</returns>
    public static Tensor<TNumber> operator /(Tensor<TNumber> left, Scalar<TNumber> right)
    {
        return left.Divide(right);
    }

    /// <summary>
    /// Divides the left operand by the right operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the division.</returns>
    public static Tensor<TNumber> operator /(Tensor<TNumber> left, Vector<TNumber> right)
    {
        return left.Divide(right);
    }

    /// <summary>
    /// Divides the left operand by the right operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the division.</returns>
    public static Tensor<TNumber> operator /(Tensor<TNumber> left, Matrix<TNumber> right)
    {
        return left.Divide(right);
    }

    /// <summary>
    /// Divides the left operand by the right operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the division.</returns>
    public static Tensor<TNumber> operator /(Tensor<TNumber> left, Tensor<TNumber> right)
    {
        return left.Divide(right);
    }

    /// <inheritdoc />
    public void Backward()
    {
        Tensor.Backward(this);
    }

    /// <inheritdoc />
    public Array ToArray()
    {
        return Tensor.ToArray(this);
    }

    /// <inheritdoc />
    public void ForceDispose()
    {
        Memory.Release(_id);
        Memory.Dispose();
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
        using var tempTensor = new Tensor<TNumber>(newHandle, Shape, newBackend, RequiresGradient);

        newHandle.CopyFromSystemMemory(Memory.ToSystemMemory());
        Memory = newHandle;
        Backend = newBackend;
        oldHandle.Dispose();

        Gradient?.To(device);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return Tensor.ToString(this);
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
        if (disposing && IsMemoryOwner && !IsGradient)
        {
            Memory.Release(_id);
            Memory.Dispose();
            Gradient?.ForceDispose();
        }
    }
}