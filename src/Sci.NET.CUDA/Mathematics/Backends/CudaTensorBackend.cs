// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory.Unmanaged;
using Sci.NET.CUDA.CuBLAS;
using Sci.NET.CUDA.Memory;
using Sci.NET.Mathematics.BLAS;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Mathematics.Tensors.Backends;

namespace Sci.NET.CUDA.Mathematics.Backends;

/// <summary>
/// A CUDA implementation of the <see cref="TensorBackend"/> interface.
/// </summary>
[PublicAPI]
public class CudaTensorBackend : TensorBackend
{
    /// <inheritdoc />
    public override INativeMemoryManager MemoryManager => new CudaMemoryManager();

    /// <inheritdoc />
    public override IBlasProvider BlasProvider => new CublasProvider();

    /// <inheritdoc />
    public override ITensor<TNumber> InnerProduct<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
    {
        throw new PlatformNotSupportedException();
    }

    /// <inheritdoc />
    public override ITensor<TNumber> ScalarMultiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
    {
        throw new PlatformNotSupportedException();
    }
}