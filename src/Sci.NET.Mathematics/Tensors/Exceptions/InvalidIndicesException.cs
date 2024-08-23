// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace Sci.NET.Mathematics.Tensors.Exceptions;

/// <summary>
/// An exception thrown when the indices are invalid.
/// </summary>
[PublicAPI]
[ExcludeFromCodeCoverage]
public class InvalidIndicesException : Exception
{
    /// <inheritdoc />
    public InvalidIndicesException(string message)
        : base(message)
    {
    }

    /// <inheritdoc />
    public InvalidIndicesException(string message, Exception inner)
        : base(message, inner)
    {
    }

    /// <summary>
    /// Throws an <see cref="InvalidIndicesException"/> if the left and right indices are not equal.
    /// </summary>
    /// <param name="left">The left indices.</param>
    /// <param name="right">The right indices.</param>
    /// <param name="message">The message to display.</param>
    /// <exception cref="InvalidIndicesException">Throws when the left and right indices are not equal.</exception>
    public static void ThrowIfNotEqual(int left, int right, string message)
    {
        if (left != right)
        {
            throw new InvalidIndicesException(message);
        }
    }
}