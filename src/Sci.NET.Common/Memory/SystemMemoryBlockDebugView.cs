// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;

namespace Sci.NET.Common.Memory;

[ExcludeFromCodeCoverage]
internal class SystemMemoryBlockDebugView<T>
    where T : unmanaged
{
    public SystemMemoryBlockDebugView(IMemoryBlock<T> block)
    {
        Items = block.Length < 10000 ? block.ToArray() : new[] { "Too many items to display" };
    }

    [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
    public Array Items { get; }
}