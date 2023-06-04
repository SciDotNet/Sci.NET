// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.Native;

/// <summary>
/// An interface exposing the CUDA Runtime API.
/// </summary>
[PublicAPI]
public interface ICudaRuntimeApi
{
    /// <summary>
    /// Allocates a block of memory on the device.
    /// </summary>
    /// <param name="count">The number of elements to allocate.</param>
    /// <typeparam name="T">The type of element stored within the memory block.</typeparam>
    /// <returns>A pointer to the memory block.</returns>
    public unsafe T* Allocate<T>(long count)
        where T : unmanaged;

    /// <summary>
    /// Frees a block of memory on the device.
    /// </summary>
    /// <param name="memoryPtr">The pointer to free.</param>
    /// <typeparam name="T">The type of element stored within the memory block.</typeparam>
    public unsafe void Free<T>(T* memoryPtr)
        where T : unmanaged;

    /// <summary>
    /// Copies a block of memory from the device to the device.
    /// </summary>
    /// <param name="source">TThe source pointer.</param>
    /// <param name="destination">The destination pointer.</param>
    /// <param name="count">The number of elements to copy.</param>
    /// <typeparam name="T">The type of element stored within the memory block.</typeparam>
    public unsafe void CopyDeviceToDevice<T>(T* source, T* destination, long count)
        where T : unmanaged;

    /// <summary>
    /// Copies a block of memory from the device to the host.
    /// </summary>
    /// <param name="source">TThe source pointer.</param>
    /// <param name="destination">The destination pointer.</param>
    /// <param name="count">The number of elements to copy.</param>
    /// <typeparam name="T">The type of element stored within the memory block.</typeparam>
    public unsafe void CopyDeviceToHost<T>(T* source, T* destination, long count)
        where T : unmanaged;

    /// <summary>
    /// Copies a block of memory from the host to the device.
    /// </summary>
    /// <param name="source">TThe source pointer.</param>
    /// <param name="destination">The destination pointer.</param>
    /// <param name="count">The number of elements to copy.</param>
    /// <typeparam name="T">The type of element stored within the memory block.</typeparam>
    public unsafe void CopyHostToDevice<T>(T* source, T* destination, long count)
        where T : unmanaged;
}