// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Sci.NET.Common.Comparison;
using Sci.NET.Common.Performance;

namespace Sci.NET.Common.Memory;

/// <summary>
/// Represents a typed handle to a memory block.
/// </summary>
/// <typeparam name="T">The type of the memory.</typeparam>
[PublicAPI]
[StructLayout(LayoutKind.Sequential)]
public readonly struct TypedMemoryHandle<T> : IValueEquatable<TypedMemoryHandle<T>>
    where T : unmanaged
{
    private readonly unsafe void* _ptr;

    /// <summary>
    /// Initializes a new instance of the <see cref="TypedMemoryHandle{T}"/> struct.
    /// </summary>
    /// <param name="memoryPtr">The pointer to the memory.</param>
    public unsafe TypedMemoryHandle(T* memoryPtr)
    {
        _ptr = memoryPtr;
    }

    /// <inheritdoc />
    [MethodImpl(ImplementationOptions.InlineOptimized)]
    public static bool operator ==(TypedMemoryHandle<T> left, TypedMemoryHandle<T> right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    [MethodImpl(ImplementationOptions.InlineOptimized)]
    public static bool operator !=(TypedMemoryHandle<T> left, TypedMemoryHandle<T> right)
    {
        return !left.Equals(right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    [MethodImpl(ImplementationOptions.InlineOptimized)]
    public override bool Equals(object? obj)
    {
        return obj is TypedMemoryHandle<T> other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    [MethodImpl(ImplementationOptions.InlineOptimized)]
    public unsafe bool Equals(TypedMemoryHandle<T> other)
    {
        return _ptr == other._ptr;
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    [MethodImpl(ImplementationOptions.InlineOptimized)]
    public override unsafe int GetHashCode()
    {
        return unchecked((int)(long)_ptr);
    }

    /// <summary>
    /// Gets the pointer to the memory.
    /// </summary>
    /// <returns>A pointer to the memory.</returns>
    public unsafe T* ToPointer()
    {
        return (T*)_ptr;
    }

    /// <summary>
    /// Gets the pointer to the memory.
    /// </summary>
    /// <typeparam name="TPtr">The type of pointer to get.</typeparam>
    /// <returns>A pointer to the memory.</returns>
    public unsafe TPtr* ToPointer<TPtr>()
        where TPtr : unmanaged
    {
        return (TPtr*)_ptr;
    }

    /// <summary>
    /// Converts the current instance to a <see cref="UIntPtr"/>.
    /// </summary>
    /// <returns>the current instance as a <see cref="UIntPtr"/>.</returns>
    public unsafe nuint ToUIntPtr()
    {
        return new nuint(_ptr);
    }

    /// <summary>
    /// Copies the memory to an array.
    /// </summary>
    /// <param name="count">The number elements to copy.</param>
    /// <returns>An array containing the data of this instance.</returns>
    public T[] CopyToArray(SizeT count)
    {
        var array = new T[count.ToInt64()];

        unsafe
        {
#pragma warning disable RCS1176
            fixed (T* ptr = &array[0])
#pragma warning restore RCS1176
            {
                NativeMemory.Copy(_ptr, ptr, new nuint((ulong)sizeof(T)) * count.ToUIntPtr());
            }
        }

        return array;
    }
}