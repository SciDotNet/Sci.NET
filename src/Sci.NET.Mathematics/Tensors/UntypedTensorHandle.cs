// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;
using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// A class containing an untyped handle to a <see cref="ITensor{TNumber}"/>.
/// </summary>
[PublicAPI]
public class UntypedTensorHandle
{
    private readonly object _handle;

    /// <summary>
    /// Initializes a new instance of the <see cref="UntypedTensorHandle"/> class.
    /// </summary>
    /// <param name="tensor">The boxed <see cref="ITensor{TNumber}"/> object.</param>
    /// <param name="shape">The <see cref="Shape"/> of the <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="device">The <see cref="IDevice"/> the <see cref="ITensor{TNumber}"/> is on.</param>
    /// <param name="handle">The handle to the memory block.</param>
    public UntypedTensorHandle(object tensor, Shape shape, IDevice device, UIntPtr handle)
    {
        _handle = tensor;
        Shape = shape;
        Device = device;
        Handle = handle;
    }

    /// <summary>
    /// Gets the <see cref="Shape"/> of the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    public Shape Shape { get; }

    /// <summary>
    /// Gets the <see cref="IDevice"/> the <see cref="ITensor{TNumber}"/> is on.
    /// </summary>
    public IDevice Device { get; }

    /// <summary>
    /// Gets the untyped handle to the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    public UIntPtr Handle { get; }

    /// <summary>
    /// Gets an untyped handle to the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to get the handle for.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>An untyped handle to the <see cref="ITensor{TNumber}"/>.</returns>
    public static unsafe UntypedTensorHandle FromTensor<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return new UntypedTensorHandle(
            tensor,
            tensor.Shape,
            tensor.Device,
            (nuint)Unsafe.AsPointer(ref Unsafe.AsRef(tensor.Handle)));
    }

    /// <summary>
    /// Gets the <see cref="ITensor{TNumber}"/> from the untyped handle.
    /// </summary>
    /// <param name="handle">The untyped handle.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The <see cref="ITensor{TNumber}"/> from the untyped handle.</returns>
    public static ITensor<TNumber> ToTensor<TNumber>(UntypedTensorHandle handle)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return (ITensor<TNumber>)handle._handle;
    }
}