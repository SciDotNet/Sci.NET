// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Sci.NET.Common.Comparison;
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
        : this(array.LongLength)
    {
        Buffer.MemoryCopy(
            Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(array)),
            _reference,
            array.LongLength * Unsafe.SizeOf<T>(),
            array.LongLength * Unsafe.SizeOf<T>());
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
        Length = count;
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
    public bool IsDisposed { get; private set; }

    /// <summary>
    /// Gets a reference to the element at the specified index.
    /// </summary>
    /// <param name="index">The index of the element.</param>
    /// <exception cref="ArgumentOutOfRangeException">The specified index was out of range.</exception>
    public unsafe ref T this[long index]
    {
        [MethodImpl(ImplementationOptions.HotPath)]
        get
        {
            if (index < 0 || index >= Length)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            return ref Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nint)index);
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
    public static bool operator !=(SystemMemoryBlock<T> left, SystemMemoryBlock<T> right)
    {
        return !left.Equals(right);
    }

    /// <summary>
    /// Fills the <see cref="SystemMemoryBlock{T}"/> with the specified values.
    /// </summary>
    /// <param name="start">The start index.</param>
    /// <param name="buffer">The values to add to the buffer.</param>
    /// <param name="bytesToCopy">The number of bytes to copy.</param>
    public unsafe void FillBytes(long start, byte[] buffer, long bytesToCopy)
    {
        var bufferPtr = Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(buffer));
        var dataPtr = Unsafe.AsPointer(ref Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nuint)start));

        Buffer.MemoryCopy(
            bufferPtr,
            dataPtr,
            Length * Unsafe.SizeOf<T>(),
            bytesToCopy);
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
    [MethodImpl(ImplementationOptions.HotPath)]
    public unsafe void Fill(T value)
    {
        if (!Vector.IsHardwareAccelerated)
        {
            goto CannotVectorize;
        }

        if (Unsafe.SizeOf<T>() > Vector<T>.Count)
        {
            goto CannotVectorize;
        }

        if (!BitOperations.IsPow2(sizeof(T)))
        {
            goto CannotVectorize;
        }

        if (Length >= Vector<T>.Count / Unsafe.SizeOf<T>())
        {
            var tmp = value;
            Vector<byte> vector;

            if (Unsafe.SizeOf<T>() == 1)
            {
                vector = new Vector<byte>(Unsafe.As<T, byte>(ref tmp));
            }
            else if (Unsafe.SizeOf<T>() == 2)
            {
                vector = (Vector<byte>)new Vector<ushort>(Unsafe.As<T, ushort>(ref tmp));
            }
            else if (Unsafe.SizeOf<T>() == 4)
            {
                vector = (typeof(T) == typeof(float))
                    ? (Vector<byte>)new Vector<float>((float)(object)tmp)
                    : (Vector<byte>)new Vector<uint>(Unsafe.As<T, uint>(ref tmp));
            }
            else if (Unsafe.SizeOf<T>() == 8)
            {
                vector = (typeof(T) == typeof(double))
                    ? (Vector<byte>)new Vector<double>((double)(object)tmp)
                    : (Vector<byte>)new Vector<ulong>(Unsafe.As<T, ulong>(ref tmp));
            }
            else if (Unsafe.SizeOf<T>() == 16)
            {
                var vec128 = Unsafe.As<T, Vector128<byte>>(ref tmp);

                switch (Vector<byte>.Count)
                {
                    case 16:
                        vector = vec128.AsVector();
                        break;
                    case 32:
                        vector = Vector256.Create(vec128, vec128).AsVector();
                        break;
                    default:
                        goto CannotVectorize;
                }
            }
            else if (Unsafe.SizeOf<T>() == 32)
            {
                if (Vector<byte>.Count == 32)
                {
                    vector = Unsafe.As<T, Vector256<byte>>(ref tmp).AsVector();
                }
                else
                {
                    goto CannotVectorize;
                }
            }
            else
            {
                goto CannotVectorize;
            }

            ref var refData = ref Unsafe.As<T, byte>(ref Unsafe.AsRef<T>(_reference));
            var totalByteLength = (nuint)(Length * Unsafe.SizeOf<T>());
            var stopLoopAtOffset = totalByteLength & (nuint)(2 * -Vector<byte>.Count);
            var offset = (nuint)0;

            if (Length >= (uint)(2 * Vector<byte>.Count / Unsafe.SizeOf<T>()))
            {
                do
                {
                    Unsafe.WriteUnaligned(ref Unsafe.AddByteOffset(ref refData, offset), vector);

                    Unsafe.WriteUnaligned(
                        ref Unsafe.AddByteOffset(ref refData, offset + (nuint)Vector<byte>.Count),
                        vector);
                    offset += (uint)(2 * Vector<byte>.Count);
                }
                while (offset < stopLoopAtOffset);
            }

            if ((totalByteLength & (nuint)Vector<byte>.Count) != 0)
            {
                Unsafe.WriteUnaligned(ref Unsafe.AddByteOffset(ref refData, offset), vector);
            }

            Unsafe.WriteUnaligned(
                ref Unsafe.AddByteOffset(ref refData, totalByteLength - (nuint)Vector<byte>.Count),
                vector);
            return;
        }

        CannotVectorize:
        long i = 0;

        if (Length >= 8)
        {
            var stopLoopAtOffset = Length & ~7;

            do
            {
                Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nint)(i * Unsafe.SizeOf<T>()) + 0) = value;
                Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nint)(i * Unsafe.SizeOf<T>()) + 1) = value;
                Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nint)(i * Unsafe.SizeOf<T>()) + 2) = value;
                Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nint)(i * Unsafe.SizeOf<T>()) + 3) = value;
                Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nint)(i * Unsafe.SizeOf<T>()) + 4) = value;
                Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nint)(i * Unsafe.SizeOf<T>()) + 5) = value;
                Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nint)(i * Unsafe.SizeOf<T>()) + 6) = value;
                Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nint)(i * Unsafe.SizeOf<T>()) + 7) = value;
            }
            while ((i += 8) < stopLoopAtOffset);
        }

        // Write next 4 elements if needed
        if ((Length & 4) != 0)
        {
            Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nint)(i * Unsafe.SizeOf<T>()) + 0) = value;
            Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nint)(i * Unsafe.SizeOf<T>()) + 1) = value;
            Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nint)(i * Unsafe.SizeOf<T>()) + 2) = value;
            Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nint)(i * Unsafe.SizeOf<T>()) + 3) = value;
            i += 4;
        }

        // Write next 2 elements if needed
        if ((Length & 2) != 0)
        {
            Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nint)(i * Unsafe.SizeOf<T>()) + 0) = value;
            Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nint)(i * Unsafe.SizeOf<T>()) + 1) = value;
            i += 2;
        }

        // Write final element if needed
        if ((Length & 1) != 0)
        {
            Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nint)(i * Unsafe.SizeOf<T>()) + 0) = value;
        }
    }

    /// <inheritdoc />
    public unsafe void CopyFromSystemMemory(SystemMemoryBlock<T> source)
    {
        if (source.Length != Length)
        {
            throw new ArgumentException("Source must have the same length as the destination.", nameof(source));
        }

        Buffer.MemoryCopy(
            source._reference,
            _reference,
            Length * Unsafe.SizeOf<T>(),
            Length * Unsafe.SizeOf<T>());
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
    public unsafe void WriteTo(Stream stream)
    {
        var reference = Unsafe.AsRef<byte>(_reference);
        var byteLength = Length * Unsafe.SizeOf<T>();

        // Write the entire block in one go if possible
        if (byteLength <= int.MaxValue)
        {
            stream.Write(new ReadOnlySpan<byte>(_reference, (int)byteLength));
            return;
        }

        // Otherwise write in chunks using long pointer
        var remaining = byteLength;
        var pointer = (byte*)Unsafe.AsPointer(ref reference);
        var offset = 0L;

        while (remaining > 0)
        {
            var chunkLength = Math.Min(remaining, int.MaxValue);
            stream.Write(new ReadOnlySpan<byte>(pointer + offset, (int)chunkLength));
            remaining -= chunkLength;
            offset += chunkLength;
        }
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
    public SystemMemoryBlock<T> ToSystemMemory()
    {
        return this;
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
        return !IsDisposed && other is not null && _reference == other._reference && Length == other.Length;
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
        if (!IsDisposed || !isDisposing)
        {
            IsDisposed = true;
            NativeMemory.Free(_reference);
        }
    }
}