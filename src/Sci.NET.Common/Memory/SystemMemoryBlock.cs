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
using Sci.NET.Common.Numerics.Intrinsics;
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
    private readonly bool _cannotDispose;

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

    private unsafe SystemMemoryBlock(T* reference, long length)
    {
        _rentals = new ConcurrentList<Guid>();
        _reference = reference;
        _cannotDispose = true;
        Length = length;
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
            ObjectDisposedException.ThrowIf(IsDisposed, this);
            ArgumentOutOfRangeException.ThrowIfLessThan(index, 0);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, Length);

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
        ObjectDisposedException.ThrowIf(left.IsDisposed, left);
        ObjectDisposedException.ThrowIf(right.IsDisposed, right);

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
        ObjectDisposedException.ThrowIf(left.IsDisposed, left);
        ObjectDisposedException.ThrowIf(right.IsDisposed, right);

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
        ObjectDisposedException.ThrowIf(IsDisposed, this);

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
        ObjectDisposedException.ThrowIf(IsDisposed, this);

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
        ObjectDisposedException.ThrowIf(IsDisposed, this);

        for (var i = 0L; i < Length; i++)
        {
            Unsafe.Add(ref Unsafe.AsRef<T>(_reference), (nuint)i) = value;
        }
    }
#pragma warning restore CA1502 // Avoid excessive complexity

    /// <inheritdoc />
    public unsafe void CopyFromSystemMemory(SystemMemoryBlock<T> source)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);

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
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        ArgumentException.ThrowIfNullOrEmpty(nameof(array));

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
        ObjectDisposedException.ThrowIf(IsDisposed, this);

        var byteLength = Length * Unsafe.SizeOf<T>();

        // Write the entire block in one go if possible
        if (byteLength <= int.MaxValue)
        {
            stream.Write(new ReadOnlySpan<byte>(_reference, (int)byteLength));
            return;
        }

        // Otherwise write in chunks using long pointer
        // Note: There is no test coverage for this code path as no in-memory stream can exceed 2.5GB.
        //       We could add a test for this by creating a custom stream that allows for a larger buffer size
        //       or by using a file stream, but this would slow down CI builds.
        var remaining = byteLength;
        var pointer = (byte*)_reference;
        var offset = 0L;

        while (remaining > 0)
        {
            var chunkLength = Math.Min(remaining, int.MaxValue - 1);
            stream.Write(new ReadOnlySpan<byte>(pointer + offset, (int)chunkLength));
            remaining -= chunkLength;
            offset += chunkLength;
        }
    }

    /// <inheritdoc />
    public unsafe void BlockCopyFrom(IMemoryBlock<T> handle, long srcIdx, long dstIdx, long count)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        ArgumentOutOfRangeException.ThrowIfLessThan(srcIdx, 0);
        ArgumentOutOfRangeException.ThrowIfLessThan(dstIdx, 0);
        ArgumentOutOfRangeException.ThrowIfLessThan(count, 0);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(srcIdx, handle.Length);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(dstIdx, Length);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(count, handle.Length - srcIdx);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(count, Length - dstIdx);

        if (handle is not SystemMemoryBlock<T> memoryBlock)
        {
            throw new InvalidOperationException($"Cannot copy from {handle.GetType().Name} to {GetType().Name}.");
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
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        ArgumentOutOfRangeException.ThrowIfLessThan(srcIdx, 0);
        ArgumentOutOfRangeException.ThrowIfLessThan(dstIdx, 0);
        ArgumentOutOfRangeException.ThrowIfLessThan(count, 0);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(srcIdx, buffer.Length);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(dstIdx, Length * Unsafe.SizeOf<T>());
        ArgumentOutOfRangeException.ThrowIfGreaterThan(count, buffer.Length - srcIdx);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(count, (Length * Unsafe.SizeOf<T>()) - dstIdx);

        var byteDestination = (byte*)ToPointer() + dstIdx;

        fixed (byte* byteSource = buffer)
        {
            Buffer.MemoryCopy(
                byteSource + srcIdx,
                byteDestination,
                count,
                count);
        }
    }

    /// <inheritdoc />
    public void UnsafeFreeMemory()
    {
        ReleaseUnmanagedResources();
        IsDisposed = true;
    }

    /// <inheritdoc />
    [MethodImpl(ImplementationOptions.HotPath)]
    public IMemoryBlock<T> Copy()
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);

        var result = new SystemMemoryBlock<T>(Length);
        CopyTo(result);
        return result;
    }

    /// <inheritdoc />
    [MethodImpl(ImplementationOptions.FastPath)]
    public SystemMemoryBlock<T> ToSystemMemory()
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);

        return this;
    }

    /// <inheritdoc />
    [MethodImpl(ImplementationOptions.FastPath)]
    public unsafe void CopyTo(IMemoryBlock<T> destination)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);

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
        ObjectDisposedException.ThrowIf(IsDisposed, this);

        return obj is SystemMemoryBlock<T> other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    [MethodImpl(ImplementationOptions.HotPath)]
    public unsafe bool Equals(SystemMemoryBlock<T>? other)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        ObjectDisposedException.ThrowIf(other?.IsDisposed ?? false, other ?? this);

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
        ObjectDisposedException.ThrowIf(IsDisposed, this);

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
        ObjectDisposedException.ThrowIf(IsDisposed, this);

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
        ObjectDisposedException.ThrowIf(IsDisposed, this);

        return ref Unsafe.AsRef<T>(_reference);
    }

    /// <summary>
    /// Gets a reference to the <see cref="SystemMemoryBlock{T}"/> at the specified index.
    /// </summary>
    /// <returns>A reference to the first element of the <see cref="SystemMemoryBlock{T}"/>.</returns>
    /// <exception cref="ObjectDisposedException">The memory block has been disposed.</exception>
    public unsafe T* ToPointer()
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);

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
        ObjectDisposedException.ThrowIf(IsDisposed, this);

        if (Length > int.MaxValue)
        {
            throw new InvalidOperationException($"Cannot create a span larger than int.MaxValue ({int.MaxValue}) elements.");
        }

        return new Span<T>(_reference, (int)Length);
    }

    /// <summary>
    /// Gets a <see cref="Span{T}"/> to the <see cref="SystemMemoryBlock{T}"/> at the specified index with the specified length.
    /// </summary>
    /// <param name="index">The index to start the span at.</param>
    /// <param name="length">The length of the span.</param>
    /// <returns>A span to this instance.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The specified index was out of range.</exception>
    public unsafe Span<T> AsSpan(long index, int length)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        ArgumentOutOfRangeException.ThrowIfLessThan(length, 0);
        ArgumentOutOfRangeException.ThrowIfLessThan(index, 0);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(index, Length - length);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, Length - index);

        return new Span<T>(_reference + index, length);
    }

    /// <summary>
    /// Reinterprets the <see cref="SystemMemoryBlock{T}"/> as a <see cref="SystemMemoryBlock{TOut}"/>.
    /// </summary>
    /// <typeparam name="TOut">The output type parameter.</typeparam>
    /// <returns>A copy of this instance as a <see cref="SystemMemoryBlock{TOut}"/>.</returns>
    /// <exception cref="ObjectDisposedException">The memory block has been disposed.</exception>
    public unsafe SystemMemoryBlock<TOut> DangerousReinterpretCast<TOut>()
        where TOut : unmanaged
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);

        return new SystemMemoryBlock<TOut>((TOut*)_reference, Length);
    }

    /// <summary>
    /// Gets a <see cref="Vector{T}"/> from the <see cref="SystemMemoryBlock{T}"/> at the specified index. This method does not perform bounds checking.
    /// </summary>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{T}"/>.</typeparam>
    /// <param name="i">The index to read from.</param>
    /// <returns>A <see cref="Vector{T}"/> from the <see cref="SystemMemoryBlock{T}"/> at the specified index.</returns>
    [MethodImpl(ImplementationOptions.FastPath)]
    public unsafe ISimdVector<TNumber> UnsafeGetVectorUnchecked<TNumber>(long i) // Must use type argument to get around INumber constraint.
        where TNumber : unmanaged, INumber<TNumber>
    {
        var span = new Span<TNumber>(_reference + i, SimdVector.Count<TNumber>());

        return SimdVector.Load(span);
    }

    /// <summary>
    /// Writes a <see cref="Vector{T}"/> to the <see cref="SystemMemoryBlock{T}"/> at the specified index. This method does not perform bounds checking.
    /// </summary>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{T}"/>.</typeparam>
    /// <param name="vector">The <see cref="Vector{T}"/> to write.</param>
    /// <param name="i">The index to write to.</param>
    public unsafe void UnsafeSetVectorUnchecked<TNumber>(ISimdVector<TNumber> vector, long i)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var span = new Span<TNumber>(_reference + i, SimdVector.Count<TNumber>());

        vector.CopyTo(span);
    }

    /// <summary>
    /// Sets the value of the <see cref="SystemMemoryBlock{T}"/> at the specified index to the vector value.
    /// </summary>
    /// <param name="vector">The vector value to set.</param>
    /// <param name="index">The index to set the value at.</param>
    /// <param name="vectorStrategy">The vector strategy to use.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ISimdVector{TNumber}"/>.</typeparam>
    public unsafe void UnsafeSetVectorUnchecked<TNumber>(ISimdVector<TNumber> vector, long index, VectorStrategy vectorStrategy)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        if (vectorStrategy == VectorStrategy.Vector)
        {
            UnsafeSetVectorUnchecked(vector, index);
        }
        else
        {
            ref var reference = ref Unsafe.AsRef<TNumber>(_reference + index);
            reference = vector[0];
        }
    }

    /// <summary>
    /// Writes a <see cref="Vector256{T}"/> to the <see cref="SystemMemoryBlock{T}"/> at the specified index.
    /// </summary>
    /// <param name="index">The index to write to.</param>
    /// <param name="vector">The <see cref="Vector256{T}"/> to write.</param>
    /// <exception cref="ArgumentOutOfRangeException">The specified index was out of range.</exception>
    public unsafe void SetVector256(long index, Vector256<T> vector)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(index, 0);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(index, Length - Vector256<T>.Count);

        Unsafe.WriteUnaligned(_reference + index, vector);
    }

    /// <inheritdoc />
    public void Rent(Guid id)
    {
        _rentals.Add(id);
    }

    /// <inheritdoc />
    public void Release(Guid id)
    {
        _ = _rentals.Remove(id);
    }

    /// <summary>
    /// Disposes the <see cref="SystemMemoryBlock{T}"/>.
    /// </summary>
    /// <param name="isDisposing">A value indicating if the instance is disposing.</param>
    private void Dispose(bool isDisposing)
    {
        ReleaseUnmanagedResources();

        if (!IsDisposed && isDisposing)
        {
            IsDisposed = true;
        }
    }

    private unsafe void ReleaseUnmanagedResources()
    {
        if (_cannotDispose || IsDisposed)
        {
            return;
        }

        NativeMemory.Free(_reference);
    }
}