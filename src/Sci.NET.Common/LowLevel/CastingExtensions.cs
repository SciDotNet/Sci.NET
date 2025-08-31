// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using Sci.NET.Common.Attributes;
using Sci.NET.Common.Performance;

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
    /// <remarks>
    /// The name is similar to the C++ reinterpret_cast, but it's not exactly the same.
    /// This is a runtime operation which is used to reinterpret the given value as another
    /// type without any C# type checking/boxing. We use it to force the compiler to treat
    /// a value as another type, which is useful for low-level operations.
    /// <b>I'm open to suggestions for a better name.</b>
    /// </remarks>
    /// <returns>The input cast to the <typeparamref name="TOut"/> type.</returns>
    /// <exception cref="InvalidOperationException">Throws if the two types are different length.</exception>
    [MethodImpl(ImplementationOptions.HotPath)]
    public static TOut ReinterpretCast<TIn, TOut>(this TIn value)
        where TIn : unmanaged
        where TOut : unmanaged
    {
        if (Unsafe.SizeOf<TIn>() != Unsafe.SizeOf<TOut>())
        {
            throw new InvalidOperationException("Cannot reinterpret cast between types of different sizes.");
        }

        return Unsafe.As<TIn, TOut>(ref value);
    }

    /// <summary>
    /// Reinterprets the given value as another type but does not check the sizes. <b>Use with caution.</b>
    /// </summary>
    /// <param name="value">The value to convert.</param>
    /// <typeparam name="TIn">The input type.</typeparam>
    /// <typeparam name="TOut">The output type.</typeparam>
    /// <remarks>
    /// The name is similar to the C++ reinterpret_cast, but it's not exactly the same.
    /// This is a runtime operation which is used to reinterpret the given value as another
    /// type without any C# type checking/boxing. We use it to force the compiler to treat
    /// a value as another type, which is useful for low-level operations.
    /// <b>I'm open to suggestions for a better name.</b>
    /// </remarks>
    /// <returns>The input cast to the <typeparamref name="TOut"/> type.</returns>
    /// <exception cref="InvalidOperationException">Throws if the two types are different length.</exception>
    [MemoryCorrupter]
    [MethodImpl(ImplementationOptions.HotPath)]
    public static TOut DangerousReinterpretCast<TIn, TOut>(this TIn value)
        where TIn : unmanaged
        where TOut : unmanaged
    {
        return Unsafe.As<TIn, TOut>(ref value);
    }
}