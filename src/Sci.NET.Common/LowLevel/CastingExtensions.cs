// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;

namespace Sci.NET.Common.LowLevel;

/// <summary>
/// A collection of casting extensions.
/// </summary>
[PublicAPI]
public static class CastingExtensions
{
    /// <summary>
    /// Reinterprets the given value as another type.
    /// </summary>
    /// <param name="value">The value to convert.</param>
    /// <typeparam name="TIn">The input type.</typeparam>
    /// <typeparam name="TOut">The output type.</typeparam>
    /// <returns>The input cast to the <typeparamref name="TOut"/> type.</returns>
    /// <exception cref="InvalidOperationException">Throws if the two types are not the same length.</exception>
    public static unsafe TOut ReinterpretCast<TIn, TOut>(this TIn value)
        where TIn : unmanaged
        where TOut : unmanaged
    {
        if (sizeof(TIn) != sizeof(TOut))
        {
            throw new InvalidOperationException("Cannot reinterpret cast between types of different sizes.");
        }

        return Unsafe.As<TIn, TOut>(ref value);
    }
}