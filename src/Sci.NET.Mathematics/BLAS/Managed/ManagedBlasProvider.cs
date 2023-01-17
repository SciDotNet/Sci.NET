// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.BLAS.Layout;

namespace Sci.NET.Mathematics.BLAS.Managed;

/// <summary>
/// A managed implementation of BLAS routines.
/// </summary>
[PublicAPI]
public class ManagedBlasProvider : IBlasProvider
{
#pragma warning disable IDE0052
    private int _maxThreads = Environment.ProcessorCount;
#pragma warning restore IDE0052

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
        var aPtr = a.ToPointer();
        var bPtr = b.ToPointer();
        var cPtr = c.ToPointer();

        var leftCopy = a.CopyToArray(24);
        var rightCopy = b.CopyToArray(12);

        _ = leftCopy;
        _ = rightCopy;

        if (transA is not TransposeType.None || transB is not TransposeType.None)
        {
            throw new PlatformNotSupportedException(
                "This implementation of BLAS does not support transposed matrices.");
        }

        for (var i = 0; i < leftDimX; i++)
        {
            for (var j = 0; j < rightDimY; j++)
            {
                var sum = T.Zero;
                for (var k = 0; k < leftDimY; k++)
                {
                    sum += *(aPtr + (i * leftDimY) + k) * *(bPtr + (k * rightDimY) + j);
                }

                *(cPtr + (i * rightDimY) + j) = (alpha * sum) + (beta * *(cPtr + (i * rightDimY) + j));
            }
        }
    }

    /// <inheritdoc />
    public void SetMaxThreads(int maxThreads)
    {
        _maxThreads = maxThreads;
    }
}