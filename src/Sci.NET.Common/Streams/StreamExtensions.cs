// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using System.Text;

namespace Sci.NET.Common.Streams;

/// <summary>
/// Provides extension methods for <see cref="Stream"/>.
/// </summary>
[PublicAPI]
public static class StreamExtensions
{
    /// <summary>
    /// Writes a value of type <typeparamref name="T"/> to the stream.
    /// </summary>
    /// <param name="stream">The stream to write to.</param>
    /// <param name="value">The value to write.</param>
    /// <typeparam name="T">The type of element to write.</typeparam>
    public static unsafe void WriteValue<T>(this Stream stream, T value)
        where T : unmanaged
    {
        var size = sizeof(T);
        var buffer = stackalloc byte[size];
        var ptr = (byte*)&value;

        for (var i = 0; i < size; i++)
        {
            buffer[i] = ptr[i];
        }

        stream.Write(new ReadOnlySpan<byte>(buffer, size));
    }

    /// <summary>
    /// Reads a value of type <typeparamref name="T"/> from the stream.
    /// </summary>
    /// <param name="stream">The stream to read from.</param>
    /// <typeparam name="T">The type of element to read.</typeparam>
    /// <returns>The element read from the stream.</returns>
    public static unsafe T ReadValue<T>(this Stream stream)
        where T : unmanaged
    {
        var size = Unsafe.SizeOf<T>();
        var value = default(T);
        var buffer = stackalloc byte[size];
        _ = stream.Read(new Span<byte>(buffer, size));
        var ptr = (byte*)&value;

        for (var i = 0; i < size; i++)
        {
            ptr[i] = buffer[i];
        }

        return value;
    }

    /// <summary>
    /// Writes a string to the stream.
    /// </summary>
    /// <param name="stream">The stream to write to.</param>
    /// <param name="value">The string to write.</param>
    /// <param name="encoding">The encoding to use.</param>
    public static void WriteString(this Stream stream, string value, Encoding? encoding = null)
    {
        encoding ??= Encoding.UTF8;
        var bytes = encoding.GetBytes(value);
        stream.WriteValue(bytes.Length);
        stream.Write(bytes);
    }

    /// <summary>
    /// Reads a string from the stream.
    /// </summary>
    /// <param name="stream">The stream to read from.</param>
    /// <param name="encoding">The encoding to use.</param>
    /// <returns>The string read from the stream.</returns>
    public static string ReadString(this Stream stream, Encoding? encoding = null)
    {
        encoding ??= Encoding.UTF8;
        var length = stream.ReadValue<int>();
        var bytes = new byte[length];
        _ = stream.Read(bytes);
        return encoding.GetString(bytes);
    }
}