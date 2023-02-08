// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;

namespace Sci.NET.Mathematics.Extensions;

/// <summary>
/// Extension methods for <see cref="SystemMemoryBlock{T}"/>.
/// </summary>
[PublicAPI]
public static class SystemMemoryBlockExtensions
{
    /// <summary>
    /// Finds the minimum value in the <see cref="IMemoryBlock{T}"/>.
    /// </summary>
    /// <param name="memoryBlock">The <see cref="IMemoryBlock{T}"/> to scan.</param>
    /// <typeparam name="TNumber">The type of number in the <see cref="IMemoryBlock{T}"/>..</typeparam>
    /// <returns>The maximum number in the <see cref="IMemoryBlock{T}"/>.</returns>
    public static TNumber Max<TNumber>(this SystemMemoryBlock<TNumber> memoryBlock)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var maximum = memoryBlock[0];

        for (var i = 1; i < memoryBlock.Length; i++)
        {
            if (memoryBlock[i] > maximum)
            {
                maximum = memoryBlock[i];
            }
        }

        return maximum;
    }

    /// <summary>
    /// Finds the minimum value in the <see cref="IMemoryBlock{T}"/>.
    /// </summary>
    /// <param name="memoryBlock">The <see cref="IMemoryBlock{T}"/> to scan.</param>
    /// <typeparam name="TNumber">The type of number in the <see cref="IMemoryBlock{T}"/>..</typeparam>
    /// <returns>The maximum number in the <see cref="IMemoryBlock{T}"/>.</returns>
    public static TNumber Min<TNumber>(this SystemMemoryBlock<TNumber> memoryBlock)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var minimum = memoryBlock[0];

        for (var i = 1; i < memoryBlock.Length; i++)
        {
            if (memoryBlock[i] < minimum)
            {
                minimum = memoryBlock[i];
            }
        }

        return minimum;
    }
}