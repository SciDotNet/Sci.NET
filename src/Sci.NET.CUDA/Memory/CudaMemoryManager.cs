// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Memory.Unmanaged;
using Sci.NET.CUDA.RuntimeApi;
using Sci.NET.CUDA.RuntimeApi.Bindings.Types;

namespace Sci.NET.CUDA.Memory;

/// <summary>
/// A CUDA implementation of <see cref="INativeMemoryManager"/>.
/// </summary>
[PublicAPI]
public class CudaMemoryManager : INativeMemoryManager
{
    /// <inheritdoc />
    public IMemoryBlock<T> Allocate<T>(SizeT count)
        where T : unmanaged
    {
        return new CudaMemoryBlock<T>(count.ToInt64());
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
        return new CudaMemoryBlock<T>(array);
    }

    /// <inheritdoc />
#pragma warning disable CA1822
    public unsafe IMemoryBlock<TNumber> CopyToHostMemory<TNumber>(
        IMemoryBlock<TNumber> tensorHandle,
        SizeT tensorElementCount)
        where TNumber : unmanaged, INumber<TNumber>
#pragma warning restore CA1822
    {
        var result = new SystemMemoryBlock<TNumber>(tensorElementCount.ToInt64());

        CudaMemoryApi.CudaMemcpy(
            result.ToPointer(),
            tensorHandle.ToPointer(),
            tensorElementCount,
            CudaMemcpyKind.DeviceToHost);

        return result;
    }
}