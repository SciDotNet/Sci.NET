// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.BLAS.Types;
using Sci.NET.Common.Memory;

namespace Sci.NET.BLAS.Api;

/// <inheritdoc />
[PublicAPI]
public class BlasProvider : IBlasProvider
{
    /// <inheritdoc />
    public unsafe TypedMemoryHandle<T> Allocate<T>(long count)
        where T : unmanaged, INumber<T>
    {
        var type = T.Zero switch
        {
            float => NumberType.Float32,
            double => NumberType.Float64,
            _ => throw new NotSupportedException($"Unsupported number type '{typeof(T)}'."),
        };

        var handle = default(nint);
        NativeMethods.AllocateMemory(count, type, ref handle);

        return new TypedMemoryHandle<T>((T*)handle.ToPointer());
    }
}