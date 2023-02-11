// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using Sci.NET.Common.Performance;

namespace Sci.NET.Common.LowLevel;

/// <summary>
/// Extension methods to reinterpret the bits of a value as another type.
/// </summary>
[PublicAPI]
public static class ReinterpretExtensions
{
    /// <summary>
    /// Reinterprets the bits of a value as another type.
    /// </summary>
    /// <param name="value">The value to reinterpret.</param>
    /// <typeparam name="TIn">The type of the input.</typeparam>
    /// <typeparam name="TOut">The type of the output.</typeparam>
    /// <returns>The bits of the input interpreted as the bits of the output.</returns>
    /// <exception cref="ArgumentException">Throws when the size of the input type
    /// is not equal to the size of the output type.</exception>
    [MethodImpl(ImplementationOptions.HotPath)]
    public static TOut Reinterpret<TIn, TOut>(this TIn value)
        where TIn : unmanaged
        where TOut : unmanaged
    {
        var size = Unsafe.SizeOf<TIn>();
        if (size != Unsafe.SizeOf<TOut>())
        {
            throw new ArgumentException("The size of the input and output types must be the same.");
        }

        var result = default(TOut);
        Unsafe.CopyBlockUnaligned(
            ref Unsafe.As<TOut, byte>(ref result),
            ref Unsafe.As<TIn, byte>(ref value),
            (uint)size);
        return result;
    }
}