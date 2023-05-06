// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;

namespace Sci.NET.Common.Memory;

/// <summary>
/// An enumerator for <see cref="SystemMemoryBlock{T}"/>.
/// </summary>
/// <typeparam name="T">The type of element in the enumeration.</typeparam>
[PublicAPI]
public ref struct SystemMemoryBlockEnumerator<T>
    where T : unmanaged
{
    private readonly SystemMemoryBlock<T> _block;
    private int _index;

    internal SystemMemoryBlockEnumerator(SystemMemoryBlock<T> block)
    {
        _block = block;
        _index = -1;
    }

    /// <summary>
    /// Gets the element at the current position of the enumerator.
    /// </summary>
    public readonly unsafe T Current => Unsafe.Add(ref Unsafe.AsRef<T>(_block.ToPointer()), _index);

    /// <summary>
    /// Advances the enumerator to the next element of the <see cref="SystemMemoryBlock{T}"/>.
    /// </summary>
    /// <returns><c>true</c> if the index was incremented, else <c>false</c>.</returns>
    public bool MoveNext()
    {
        var index = _index + 1;

        if (index >= _block.Length)
        {
            return false;
        }

        _index = index;
        return true;
    }
}