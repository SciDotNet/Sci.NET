// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Common.Memory.Unmanaged;

/// <summary>
/// The default native memory provider.
/// </summary>
[PublicAPI]
public class DefaultNativeMemoryManager : INativeMemoryManager
{
    /// <inheritdoc />
    public IMemoryBlock<T> Allocate<T>(SizeT count)
        where T : unmanaged
    {
        return new SystemMemoryBlock<T>(count.ToInt64());
    }

    /// <inheritdoc />
    public void Free<T>(IMemoryBlock<T> handle)
        where T : unmanaged
    {
        handle.Dispose();
    }

    /// <inheritdoc />
    public void CopyTo<T>(IMemoryBlock<T> source, IMemoryBlock<T> destination)
        where T : unmanaged
    {
        source.CopyTo(destination);
    }

    /// <inheritdoc />
    public IMemoryBlock<T> CopyFromArray<T>(T[] array)
        where T : unmanaged
    {
        return new SystemMemoryBlock<T>(array);
    }

    /// <inheritdoc />
    public IMemoryBlock<TNumber> CopyToHostMemory<TNumber>(
        IMemoryBlock<TNumber> tensorHandle,
        SizeT tensorElementCount)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return tensorHandle;
    }
}