// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Sci.NET.Common.Memory;

namespace Sci.NET.Common.Extensions;

/// <summary>
/// Extension methods for <see cref="Stream"/>.
/// </summary>
[PublicAPI]
public static class StreamExtensions
{
    /// <summary>
    /// Reads a value of type <typeparamref name="T"/> from a <see cref="Stream"/>.
    /// </summary>
    /// <param name="stream">The <see cref="Stream"/> to write to.</param>
    /// <param name="value">The value to write.</param>
    /// <typeparam name="T">The type of element to write.</typeparam>
    public static unsafe void Write<T>(this Stream stream, T value)
        where T : unmanaged
    {
        var size = Marshal.SizeOf<T>();
        var buffer = new byte[size];

        Buffer.MemoryCopy(
            Unsafe.AsPointer(ref value),
            Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(buffer)),
            size,
            size);

        stream.Write(buffer, 0, size);
    }

    /// <summary>
    /// Reads an array of type <typeparamref name="T"/> from a <see cref="Stream"/>.
    /// </summary>
    /// <param name="stream">The <see cref="Stream"/> to write to.</param>
    /// <param name="array">The array of values to write.</param>
    /// <typeparam name="T">The type of element to write.</typeparam>
    public static unsafe void Write<T>(this Stream stream, T[] array)
        where T : unmanaged
    {
        var size = Marshal.SizeOf<T>();
        var buffer = new byte[size * array.Length];

        Buffer.MemoryCopy(
            Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(array)),
            Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(buffer)),
            buffer.LongLength,
            buffer.LongLength);

        stream.Write(buffer, 0, size * array.Length);
    }

    /// <summary>
    /// Reads a value of type <typeparamref name="T"/> from a <see cref="Stream"/>.
    /// </summary>
    /// <param name="stream">The stream to read from.</param>
    /// <typeparam name="T">The type of element to write.</typeparam>
    /// <returns>The value read from the <paramref name="stream"/>.</returns>
    /// <exception cref="EndOfStreamException">The stream ended before the values could be read.</exception>
    public static T Read<T>(this Stream stream)
        where T : unmanaged
    {
        var size = Marshal.SizeOf<T>();
        var buffer = new byte[size];

        return stream.Read(buffer, 0, size) != size
            ? throw new EndOfStreamException("The stream ended before the value could be read.")
            : Unsafe.As<byte, T>(ref buffer[0]);
    }

    /// <summary>
    /// Reads an array of type <typeparamref name="T"/> from a <see cref="Stream"/>.
    /// </summary>
    /// <param name="stream">The stream to read from.</param>
    /// <param name="count">The number of elements to read.</param>
    /// <typeparam name="T">The type of element to write.</typeparam>
    /// <returns>The values read from the <paramref name="stream"/>.</returns>
    public static SystemMemoryBlock<T> Read<T>(this Stream stream, long count)
        where T : unmanaged
    {
        var block = new SystemMemoryBlock<T>(count);
        var size = Marshal.SizeOf<T>();
        var buffer = new byte[4096];

        for (var i = 0L; i < count * size;)
        {
            var start = i;
            var startStreamPosition = stream.Position;
            var bytesRead = stream.Read(buffer);
            var offsetCount = bytesRead < count * size ? bytesRead : count * size;

            block.FillBytes(start, buffer, offsetCount);

            if (bytesRead == 0)
            {
                break;
            }

            i += offsetCount;
            stream.Position = startStreamPosition + offsetCount;
        }

        return block;
    }
}