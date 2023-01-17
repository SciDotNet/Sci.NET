// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.InteropServices;

namespace Sci.NET.Common.Memory.Unmanaged;

/// <summary>
/// The default native memory provider.
/// </summary>
[PublicAPI]
public class DefaultNativeMemoryManager : INativeMemoryManager
{
    /// <inheritdoc />
    public unsafe TypedMemoryHandle<T> Allocate<T>(SizeT count)
        where T : unmanaged
    {
        var bytes = new nuint((ulong)sizeof(T) * (ulong)count.ToInt64());
        var handle = (T*)NativeMemory.AllocZeroed(bytes, (nuint)sizeof(T));
        return new TypedMemoryHandle<T>(handle);
    }

    /// <inheritdoc />
    public unsafe void Free<T>(TypedMemoryHandle<T> handle)
        where T : unmanaged
    {
        NativeMemory.Free(handle.ToPointer());
    }

    /// <inheritdoc />
    public unsafe void Copy<T>(TypedMemoryHandle<T> source, TypedMemoryHandle<T> destination, SizeT count)
        where T : unmanaged
    {
        NativeMemory.Copy(source.ToPointer(), destination.ToPointer(), (nuint)(sizeof(T) * count.ToInt64()));
    }

    /// <inheritdoc />
    public unsafe TypedMemoryHandle<T> CopyFromArray<T>(T[] array)
        where T : unmanaged
    {
        var handle = Allocate<T>(array.Length);

#pragma warning disable RCS1176
        fixed (T* ptr = &array[0])
#pragma warning restore RCS1176
        {
            Copy(new TypedMemoryHandle<T>(ptr), handle, array.Length);
        }

        return handle;
    }

    /// <inheritdoc />
    public TypedMemoryHandle<TNumber> CopyToHostMemory<TNumber>(
        TypedMemoryHandle<TNumber> tensorHandle,
        SizeT tensorElementCount)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return tensorHandle;
    }
}