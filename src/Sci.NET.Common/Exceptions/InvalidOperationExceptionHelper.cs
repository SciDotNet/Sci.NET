// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace Sci.NET.Common.Exceptions;

/// <summary>
/// Provides helper methods to throw exceptions.
/// </summary>
public static class InvalidOperationExceptionHelper
{
    /// <summary>
    /// Throws an <see cref="InvalidOperationException"/> with a message that the object is disposed.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <param name="expected">The other value to compare.</param>
    /// <param name="customMessage">The custom message to include in the exception.</param>
    /// <typeparam name="T">The type of the value.</typeparam>
    /// <exception cref="InvalidOperationException">The value of <paramref name="value"/> is not equal to the <paramref name="expected"/> value.</exception>
    [StackTraceHidden]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    [ExcludeFromCodeCoverage]
    public static void ThrowIfNotEqual<T>(T value, T expected, string customMessage)
        where T : IEquatable<T>
    {
        if (!value.Equals(expected))
        {
            throw new InvalidOperationException(customMessage);
        }
    }
}