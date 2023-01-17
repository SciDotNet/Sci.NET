// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.CUDA.CuBLAS.Api;
using Sci.NET.CUDA.CuBLAS.Types;
using Sci.NET.Mathematics.BLAS;
using Sci.NET.Mathematics.BLAS.Layout;

namespace Sci.NET.CUDA.CuBLAS;

/// <summary>
/// A CUBLAS implementation of <see cref="IBlasProvider"/>.
/// </summary>
[PublicAPI]
public class CublasProvider : IBlasProvider
{
    /// <inheritdoc />
    public unsafe void Gemm<T>(
        TransposeType transA,
        TransposeType transB,
        int leftDimX,
        int rightDimY,
        int leftDimY,
        T alpha,
        TypedMemoryHandle<T> a,
        int lda,
        TypedMemoryHandle<T> b,
        int ldb,
        T beta,
        TypedMemoryHandle<T> c,
        int ldc)
        where T : unmanaged, INumber<T>
    {
        switch (T.One)
        {
            case float:
                var alphaFloat = *(float*)&alpha;
                var betaFloat = *(float*)&beta;

                CublasApi.Shared.Float32GemmV2(
                    CublasTransposeType.CublasOpN,
                    CublasTransposeType.CublasOpN,
                    leftDimX,
                    rightDimY,
                    leftDimY,
                    ref alphaFloat,
                    a.ToPointer<float>(),
                    leftDimY,
                    b.ToPointer<float>(),
                    rightDimY,
                    ref betaFloat,
                    c.ToPointer<float>(),
                    leftDimX);
                break;
            default:
                throw new PlatformNotSupportedException(
                    $"The CuBLAS Provider does not support the type '{typeof(T)}'.");
        }
    }

    /// <inheritdoc />
    public void SetMaxThreads(int maxThreads)
    {
        throw new PlatformNotSupportedException("The CuBLAS Provider does not support setting the maximum number of threads.");
    }
}