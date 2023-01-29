// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Sci.NET.Common.Comparison;
using Sci.NET.Common.Memory.ReferenceCounting;
using Sci.NET.Common.Performance;

namespace Sci.NET.Common.Memory;

/// <summary>
/// A <see cref="IMemoryBlock{T}"/> implementation for system memory.
/// </summary>
/// <typeparam name="T">The type of elements stored in the <see cref="IMemoryBlock{T}"/>.</typeparam>
[PublicAPI]
[DebuggerDisplay("{ToString(),raw}")]
[DebuggerTypeProxy(typeof(SystemMemoryBlockDebugView<>))]
public sealed class SystemMemoryBlock<T> : IMemoryBlock<T>, IEquatable<SystemMemoryBlock<T>>
    where T : unmanaged
{
    private readonly unsafe T* _reference;

    /// <summary>
    /// Initializes a new instance of the <see cref="SystemMemoryBlock{T}"/> class.
    /// </summary>
    /// <param name="array">The array to create the span from.</param>
    public unsafe SystemMemoryBlock(T[] array)
    {
        if (!typeof(T).IsValueType || array.GetType() != typeof(T[]))
        {
            throw new ArgumentException("The array must be of type T[].");
        }

        _reference = (T*)Unsafe.AsPointer(ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(array), 0));
        ReferenceCount = new ReferenceCount();
        ReferenceCount.Increment();
        Length = array.Length;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SystemMemoryBlock{T}"/> class.
    /// </summary>
    /// <param name="count">The number of elements to allocate.</param>
    public unsafe SystemMemoryBlock(long count)
    {
        var length = (nuint)count;
        var elementSize = (nuint)Unsafe.SizeOf<T>();
        var totalSize = length * elementSize;

        _reference = (T*)NativeMemory.AllocZeroed(totalSize);
        ReferenceCount = new ReferenceCount();
        ReferenceCount.Increment();
        Length = count;
    }

    private unsafe SystemMemoryBlock(T* reference, long start, long length)
    {
        _reference = (T*)Unsafe.AsPointer(ref Unsafe.Add(ref Unsafe.AsRef<T>(reference), (nuint)start));
        ReferenceCount = new ReferenceCount();
        ReferenceCount.Increment();
        Length = length;
    }

    /// <summary>
    /// Finalizes an instance of the <see cref="SystemMemoryBlock{T}"/> class.
    /// </summary>
    ~SystemMemoryBlock()
    {
        Dispose(false);
    }

    /// <inheritdoc />
    public long Length { get; }

    /// <inheritdoc />
    public ReferenceCount ReferenceCount { get; }

    /// <inheritdoc />
    public unsafe ref T this[long index]
    {
        [MethodImpl(ImplementationOptions.HotPath)]
        get
        {
            if (index < 0 || index >= Length)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(index),
                    "Index was out of range. Must be non-negative and less than the size of the collection.");
            }

            ref var result = ref Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nuint)index);

            return ref result;
        }
    }

    /// <summary>
    /// Determines if the left and right <see cref="SystemMemoryBlock{T}"/>s are equal.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>A value indicating whether the two operands are equal.</returns>
    public static bool operator ==(SystemMemoryBlock<T> left, SystemMemoryBlock<T> right)
    {
        return left.Equals(right);
    }

    /// <summary>
    /// Determines if the left and right <see cref="SystemMemoryBlock{T}"/>s are not equal.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>A value indicating whether the two operands are equal.</returns>
    public static unsafe bool operator !=(SystemMemoryBlock<T> left, SystemMemoryBlock<T> right)
    {
        return left._reference != right._reference || left.Length != right.Length;
    }

    /// <inheritdoc />
    [MethodImpl(ImplementationOptions.FastPath)]
    public unsafe void Fill(T value)
    {
        nuint i = 0;
        var reference = Unsafe.AsRef<T>(_reference);

        if (Length >= 8)
        {
            var stopLoopAt = (nuint)(Length - 8);

            do
            {
                Unsafe.Add(ref reference, i + 0) = value;
                Unsafe.Add(ref reference, i + 1) = value;
                Unsafe.Add(ref reference, i + 2) = value;
                Unsafe.Add(ref reference, i + 3) = value;
                Unsafe.Add(ref reference, i + 4) = value;
                Unsafe.Add(ref reference, i + 5) = value;
                Unsafe.Add(ref reference, i + 6) = value;
                Unsafe.Add(ref reference, i + 7) = value;
            }
            while ((i += 8) <= stopLoopAt);
        }

        if ((Length & 4) != 0)
        {
            Unsafe.Add(ref reference, i + 0) = value;
            Unsafe.Add(ref reference, i + 1) = value;
            Unsafe.Add(ref reference, i + 2) = value;
            Unsafe.Add(ref reference, i + 3) = value;
        }

        if ((Length & 2) != 0)
        {
            Unsafe.Add(ref reference, i + 0) = value;
            Unsafe.Add(ref reference, i + 1) = value;
        }

        if ((Length & 1) != 0)
        {
            Unsafe.Add(ref reference, i + 0) = value;
        }
    }

    /// <inheritdoc />
    public unsafe IMemoryBlock<T> Slice(long start, long length)
    {
        return new SystemMemoryBlock<T>(_reference, start, length);
    }

    /// <inheritdoc />
    [MethodImpl(ImplementationOptions.HotPath)]
    public unsafe T[] ToArray()
    {
        if (Length == 0)
        {
            return Array.Empty<T>();
        }

        var result = new T[Length];

        Buffer.MemoryCopy(
            _reference,
            Unsafe.AsPointer(ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(result), 0)),
            Length * Unsafe.SizeOf<T>(),
            Length * Unsafe.SizeOf<T>());

        return result;
    }

    /// <inheritdoc />
    public unsafe void CopyFrom(T[] array)
    {
        if (array is null)
        {
            throw new ArgumentNullException(nameof(array));
        }

        if (array.Length != Length)
        {
            throw new ArgumentException("Array must have the same length as the source.", nameof(array));
        }

        Buffer.MemoryCopy(
            Unsafe.AsPointer(ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(array), 0)),
            _reference,
            Length * Unsafe.SizeOf<T>(),
            Length * Unsafe.SizeOf<T>());
    }

    /// <inheritdoc />
    [MethodImpl(ImplementationOptions.HotPath)]
    public IMemoryBlock<T> Copy()
    {
        var result = new SystemMemoryBlock<T>(Length);
        CopyTo(result);
        return result;
    }

    /// <inheritdoc />
    [MethodImpl(ImplementationOptions.FastPath)]
    public unsafe void CopyTo(IMemoryBlock<T> destination)
    {
        if (destination is not SystemMemoryBlock<T> systemMemoryBlock)
        {
            throw new ArgumentException($"Destination must be a {nameof(SystemMemoryBlock<T>)}.", nameof(destination));
        }

        if (destination.Length != Length)
        {
            throw new ArgumentException("Destination must have the same length as the source.", nameof(destination));
        }

        Buffer.MemoryCopy(
            _reference,
            systemMemoryBlock._reference,
            Length * Unsafe.SizeOf<T>(),
            Length * Unsafe.SizeOf<T>());
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    [MethodImpl(ImplementationOptions.HotPath)]
    public override bool Equals(object? obj)
    {
        return obj is SystemMemoryBlock<T> other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    [MethodImpl(ImplementationOptions.HotPath)]
    public unsafe bool Equals(SystemMemoryBlock<T>? other)
    {
        return other is not null && _reference == other._reference && Length == other.Length;
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    [MethodImpl(ImplementationOptions.HotPath)]
    public override unsafe int GetHashCode()
    {
        return HashCode.Combine((int)((long)_reference & uint.MaxValue), (int)((long)_reference >> 32));
    }

    /// <summary>
    /// For <see cref="Span{Char}"/>, returns a string representation of the <see cref="SystemMemoryBlock{T}"/>,
    /// otherwise, returns the name of the type and the length of the <see cref="SystemMemoryBlock{T}"/>.
    /// </summary>
    /// <returns>
    /// A string representation of the <see cref="SystemMemoryBlock{T}"/>,
    /// otherwise, the name of the type and the length of the <see cref="SystemMemoryBlock{T}"/>.
    /// </returns>
    public override unsafe string ToString()
    {
        return typeof(T) == typeof(char)
            ? new string(new ReadOnlySpan<char>(_reference, checked((int)Length)))
            : $"{nameof(SystemMemoryBlock<T>)}<{typeof(T).Name}>[{Length}]";
    }

    /// <summary>
    /// Returns an enumerator that iterates through the collection.
    /// </summary>
    /// <returns>An enumerator that iterates over the collection.</returns>
    public SystemMemoryBlockEnumerator<T> GetEnumerator()
    {
        return new SystemMemoryBlockEnumerator<T>(this);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Gets a reference to the <see cref="SystemMemoryBlock{T}"/>.
    /// </summary>
    /// <returns>A reference to the <see cref="SystemMemoryBlock{T}"/>.</returns>
    public unsafe ref T GetReference()
    {
        return ref Unsafe.AsRef<T>(_reference);
    }

    /// <summary>
    /// Gets a reference to the <see cref="SystemMemoryBlock{T}"/> at the specified index.
    /// </summary>
    /// <returns>A reference to the first element of the <see cref="SystemMemoryBlock{T}"/>.</returns>
    public unsafe T* ToPointer()
    {
        return _reference;
    }

    /// <summary>
    /// Disposes the <see cref="SystemMemoryBlock{T}"/>.
    /// </summary>
    /// <param name="isDisposing">A value indicating if the instance is disposing.</param>
    private unsafe void Dispose(bool isDisposing)
    {
        ReferenceCount.Decrement();

        if (ReferenceCount.IsZero() || !isDisposing)
        {
            NativeMemory.Free(_reference);
        }
    }
}