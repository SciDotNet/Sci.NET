// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using Sci.NET.Common.Memory;

namespace Sci.NET.Test.Common.Assertions.Memory;

/// <summary>
/// Extension methods for <see cref="IMemoryBlock{TCollection}"/> assertions.
/// </summary>
[PublicAPI]
[ExcludeFromCodeCoverage]
[DebuggerNonUserCode]
public static class MemoryBlockAssertionExtensions
{
    /// <summary>
    /// Assertions for <see cref="IMemoryBlock{TCollection}"/>.
    /// </summary>
    /// <param name="memoryBlock">The target memory block.</param>
    /// <typeparam name="TCollection">The type of element in the collection.</typeparam>
    /// <returns>A <see cref="MemoryBlockAssertions{TCollection}"/> instance.</returns>
    public static MemoryBlockAssertions<TCollection> Should<TCollection>(this IMemoryBlock<TCollection> memoryBlock)
        where TCollection : unmanaged
    {
        return new MemoryBlockAssertions<TCollection>(memoryBlock);
    }
}