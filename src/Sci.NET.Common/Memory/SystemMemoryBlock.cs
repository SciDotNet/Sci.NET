// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Sci.NET.Common.Collections;
using Sci.NET.Common.Comparison;
using Sci.NET.Common.Performance;

namespace Sci.NET.Common.Memory;

/// <summary>
/// A <see cref="IMemoryBlock{T}"/> implementation for system memory.
/// </summary>
/// <typeparam name="T">The type of elements stored in the <see cref="IMemoryBlock{T}"/>.</typeparam>
[DebuggerDisplay("{ToString(),raw}")]
[DebuggerTypeProxy(typeof(SystemMemoryBlockDebugView<>))]
[PublicAPI]
public sealed class SystemMemoryBlock<T> : IMemoryBlock<T>, IEquatable<SystemMemoryBlock<T>>
    where T : unmanaged
{
    private readonly unsafe T* _reference;
    private readonly ConcurrentList<Guid> _rentals;

    /// <summary>
    /// Initializes a new instance of the <see cref="SystemMemoryBlock{T}"/> class.
    /// </summary>
    /// <param name="array">The array to create the span from.</param>
    public unsafe SystemMemoryBlock(T[] array)
        : this(array.LongLength)
    {
        _rentals = new ConcurrentList<Guid>();

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
        _rentals = new ConcurrentList<Guid>();

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
        if (!IsDisposed)
        {
            Dispose(false);
        }
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
    /// <exception cref="ObjectDisposedException">Throws when the memory block has been disposed.</exception>
    public unsafe ref T this[long index]
    {
        [MethodImpl(ImplementationOptions.HotPath)]
        get
        {
            if (IsDisposed)
            {
                throw new ObjectDisposedException("The memory block has been disposed.");
            }

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
        if (left.IsDisposed || right.IsDisposed)
        {
            return false;
        }

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
        if (left.IsDisposed || right.IsDisposed)
        {
            return true;
        }

        return !left.Equals(right);
    }

    /// <summary>
    /// Fills the <see cref="SystemMemoryBlock{T}"/> with the specified values.
    /// </summary>
    /// <param name="start">The start index.</param>
    /// <param name="buffer">The values to add to the buffer.</param>
    /// <param name="bytesToCopy">The number of bytes to copy.</param>
    /// <exception cref="ObjectDisposedException">Throws when the object has already been disposed.</exception>
    public unsafe void FillBytes(long start, byte[] buffer, long bytesToCopy)
    {
        if (IsDisposed)
        {
            throw new ObjectDisposedException("The memory block has been disposed.");
        }

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
        if (IsDisposed)
        {
            throw new ObjectDisposedException("The memory block has been disposed.");
        }

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

#pragma warning disable CA1502 // Avoid excessive complexity
    /// <inheritdoc />
    [MethodImpl(ImplementationOptions.HotPath)]
    [SuppressMessage("ReSharper", "CyclomaticComplexity", Justification = "Performance critical code.")]
    public unsafe void Fill(T value)
    {
        if (IsDisposed)
        {
            throw new ObjectDisposedException("The memory block has been disposed.");
        }

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
#pragma warning restore CA1502 // Avoid excessive complexity

    /// <inheritdoc />
    public unsafe void CopyFromSystemMemory(SystemMemoryBlock<T> source)
    {
        if (IsDisposed)
        {
            throw new ObjectDisposedException("The memory block has been disposed.");
        }

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
        if (IsDisposed)
        {
            throw new ObjectDisposedException("The memory block has been disposed.");
        }

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
        if (IsDisposed)
        {
            throw new ObjectDisposedException("The memory block has been disposed.");
        }

        var byteLength = Length * Unsafe.SizeOf<T>();

        // Write the entire block in one go if possible
        if (byteLength <= int.MaxValue)
        {
            stream.Write(new ReadOnlySpan<byte>(_reference, (int)byteLength));
            return;
        }

        // Otherwise write in chunks using long pointer
        var remaining = byteLength;
        var pointer = (byte*)_reference;
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
    public unsafe void BlockCopyFrom(IMemoryBlock<T> handle, long srcIdx, long dstIdx, long count)
    {
        if (IsDisposed)
        {
            throw new ObjectDisposedException("The memory block has been disposed.");
        }

        if (handle is not SystemMemoryBlock<T> memoryBlock)
        {
            throw new InvalidOperationException($"Cannot copy from {handle.GetType().Name} to {GetType().Name}.");
        }

        if (srcIdx < 0 || srcIdx >= memoryBlock.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(srcIdx),
                "Source index must be within the bounds of the source block.");
        }

        if (dstIdx < 0 || dstIdx >= Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(dstIdx),
                "Destination index must be within the bounds of the destination block.");
        }

        if (count < 0 || srcIdx + count > memoryBlock.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(count),
                "The number of elements to copy must be within the bounds of the source block.");
        }

        if (dstIdx + count > Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(count),
                "The number of elements to copy must be within the bounds of the destination block.");
        }

        Buffer.MemoryCopy(
            memoryBlock._reference + srcIdx,
            _reference + dstIdx,
            count * Unsafe.SizeOf<T>(),
            count * Unsafe.SizeOf<T>());
    }

    /// <inheritdoc />
    public unsafe void BlockCopyFrom(Span<byte> buffer, int srcIdx, int dstIdx, int count)
    {
        if (IsDisposed)
        {
            throw new ObjectDisposedException("The memory block has been disposed.");
        }

        if (srcIdx < 0 || srcIdx >= buffer.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(srcIdx),
                "Source index must be within the bounds of the source buffer.");
        }

        if (dstIdx < 0 || dstIdx >= Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(dstIdx),
                "Destination index must be within the bounds of the destination block.");
        }

        if (count < 0 || srcIdx + count > buffer.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(count),
                "The number of elements to copy must be within the bounds of the source buffer.");
        }

        if (dstIdx + count > Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(count),
                "The number of elements to copy must be within the bounds of the destination block.");
        }

        var byteCount = count * Unsafe.SizeOf<T>();
        var byteOffset = srcIdx * Unsafe.SizeOf<T>();
        var byteDestination = _reference + dstIdx;

        fixed (byte* byteSource = buffer)
        {
            Buffer.MemoryCopy(
                byteSource + byteOffset,
                byteDestination,
                byteCount,
                byteCount);
        }
    }

    /// <inheritdoc />
    [MethodImpl(ImplementationOptions.HotPath)]
    public IMemoryBlock<T> Copy()
    {
        if (IsDisposed)
        {
            throw new ObjectDisposedException("The memory block has been disposed.");
        }

        var result = new SystemMemoryBlock<T>(Length);
        CopyTo(result);
        return result;
    }

    /// <inheritdoc />
    [MethodImpl(ImplementationOptions.FastPath)]
    public SystemMemoryBlock<T> ToSystemMemory()
    {
        if (IsDisposed)
        {
            throw new ObjectDisposedException("The memory block has been disposed.");
        }

        return this;
    }

    /// <inheritdoc />
    [MethodImpl(ImplementationOptions.FastPath)]
    public unsafe void CopyTo(IMemoryBlock<T> destination)
    {
        if (IsDisposed)
        {
            throw new ObjectDisposedException("The memory block has been disposed.");
        }

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
        if (IsDisposed)
        {
            return false;
        }

        return obj is SystemMemoryBlock<T> other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    [MethodImpl(ImplementationOptions.HotPath)]
    public unsafe bool Equals(SystemMemoryBlock<T>? other)
    {
        if (IsDisposed)
        {
            return false;
        }

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
        if (IsDisposed)
        {
            return "SystemMemoryBlock disposed.";
        }

        return typeof(T) == typeof(char)
            ? new string(new ReadOnlySpan<char>(_reference, checked((int)Length)))
            : $"{nameof(SystemMemoryBlock<T>)}<{typeof(T).Name}>[{Length}]";
    }

    /// <summary>
    /// Returns an enumerator that iterates through the collection.
    /// </summary>
    /// <returns>An enumerator that iterates over the collection.</returns>
    /// <exception cref="ObjectDisposedException">The memory block has been disposed.</exception>
    public SystemMemoryBlockEnumerator<T> GetEnumerator()
    {
        if (IsDisposed)
        {
            throw new ObjectDisposedException("The memory block has been disposed.");
        }

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
    /// <exception cref="ObjectDisposedException">The memory block has been disposed.</exception>
    public unsafe ref T GetReference()
    {
        if (IsDisposed)
        {
            throw new ObjectDisposedException("The memory block has been disposed.");
        }

        return ref Unsafe.AsRef<T>(_reference);
    }

    /// <summary>
    /// Gets a reference to the <see cref="SystemMemoryBlock{T}"/> at the specified index.
    /// </summary>
    /// <returns>A reference to the first element of the <see cref="SystemMemoryBlock{T}"/>.</returns>
    /// <exception cref="ObjectDisposedException">The memory block has been disposed.</exception>
    public unsafe T* ToPointer()
    {
        if (IsDisposed)
        {
            throw new ObjectDisposedException("The memory block has been disposed.");
        }

        return _reference;
    }

    /// <summary>
    /// Gets a <see cref="Span{T}"/> to the <see cref="SystemMemoryBlock{T}"/>.
    /// </summary>
    /// <returns>A span to this instance.</returns>
    /// <exception cref="InvalidOperationException">The length of the <see cref="SystemMemoryBlock{T}"/> is too big to create a <see cref="Span{T}"/>.</exception>
    /// <exception cref="ObjectDisposedException">The memory block has been disposed.</exception>
    public unsafe Span<T> AsSpan()
    {
        if (IsDisposed)
        {
            throw new ObjectDisposedException("The memory block has been disposed.");
        }

        if (Length > int.MaxValue)
        {
            throw new InvalidOperationException("Cannot create a span larger than int.MaxValue");
        }

        return new Span<T>(_reference, (int)Length);
    }

    /// <summary>
    /// Reinterprets the <see cref="SystemMemoryBlock{T}"/> as a <see cref="SystemMemoryBlock{TOut}"/>.
    /// </summary>
    /// <typeparam name="TOut">The output type parameter.</typeparam>
    /// <returns>A copy of this instance as a <see cref="SystemMemoryBlock{TOut}"/>.</returns>
    /// <exception cref="InvalidOperationException">Throws when the sizes of each type are incompatible.</exception>
    /// <exception cref="ObjectDisposedException">The memory block has been disposed.</exception>
    public unsafe SystemMemoryBlock<TOut> DangerousReinterpretCast<TOut>()
        where TOut : unmanaged
    {
        if (IsDisposed)
        {
            throw new ObjectDisposedException("The memory block has been disposed.");
        }

        var totalBytes = Length * Unsafe.SizeOf<T>();
        var tOutSize = Unsafe.SizeOf<TOut>();

        if (totalBytes % tOutSize != 0)
        {
            throw new InvalidOperationException("Cannot reinterpret cast to a type with a different size");
        }

        var resultLength = totalBytes / tOutSize;
        var result = new SystemMemoryBlock<TOut>(resultLength);

        Buffer.MemoryCopy(
            _reference,
            result._reference,
            totalBytes,
            totalBytes);

        return result;
    }

    /// <summary>
    /// Gets a <see cref="Vector256{T}"/> from the <see cref="SystemMemoryBlock{T}"/> at the specified index.
    /// </summary>
    /// <param name="i">The index to read from.</param>
    /// <returns>A <see cref="Vector256{T}"/> from the <see cref="SystemMemoryBlock{T}"/> at the specified index.</returns>
    public unsafe Vector256<T> GetVector256(int i)
    {
        return Unsafe.ReadUnaligned<Vector256<T>>(_reference + i);
    }

    /// <summary>
    /// Writes a <see cref="Vector256{T}"/> to the <see cref="SystemMemoryBlock{T}"/> at the specified index.
    /// </summary>
    /// <param name="index">The index to write to.</param>
    /// <param name="vector">The <see cref="Vector256{T}"/> to write.</param>
    public unsafe void SetVector256(long index, Vector256<T> vector)
    {
        Unsafe.WriteUnaligned(_reference + index, vector);
    }

    /// <param name="id"></param>
    /// <inheritdoc />
    public void Rent(Guid id)
    {
        _rentals.Add(id);
    }

    /// <param name="id"></param>
    /// <inheritdoc />
    public void Release(Guid id)
    {
        _ = _rentals.Remove(id);
    }

    /// <summary>
    /// Disposes the <see cref="SystemMemoryBlock{T}"/>.
    /// </summary>
    /// <param name="isDisposing">A value indicating if the instance is disposing.</param>
    private unsafe void Dispose(bool isDisposing)
    {
        ReleaseUnmanagedResources();

        if (!IsDisposed && isDisposing)
        {
            IsDisposed = true;
        }
    }

    private unsafe void ReleaseUnmanagedResources()
    {
        NativeMemory.Free(_reference);
    }
}