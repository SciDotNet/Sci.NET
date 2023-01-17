// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;
using Sci.NET.CUDA.CuBLAS.Types;

namespace Sci.NET.CUDA.CuBLAS.Api;

/// <summary>
/// Provides an API for the cuBLAS library.
/// </summary>
[PublicAPI]
public sealed class CublasApi : IDisposable
{
#pragma warning disable CA1045
    private readonly OpaqueCublasHandle _handle;

    /// <summary>
    /// Initializes a new instance of the <see cref="CublasApi"/> class.
    /// </summary>
    public CublasApi()
    {
        _handle = Create();
    }

    /// <summary>
    /// Finalizes an instance of the <see cref="CublasApi"/> class.
    /// </summary>
    ~CublasApi()
    {
        ReleaseUnmanagedResources();
    }

    /// <summary>
    /// Gets the shared CuBLAS handle.
    /// </summary>
    public static CublasApi Shared { get; } = new();

    /// <summary>
    /// Creates a cuBLAS handle.
    /// </summary>
    /// <returns>An opaque cuBLAS handle.</returns>
    public static OpaqueCublasHandle Create()
    {
        return CublasNativeMethods.Create();
    }

    /// <summary>
    /// Performs a single precision matrix-matrix multiplication.
    /// </summary>
    /// <param name="transA">A operand operation mode.</param>
    /// <param name="transB">B operand operation mode.</param>
    /// <param name="m">Number of rows of matrix A and matrix C.</param>
    /// <param name="n">Number of columns in matrix B and matrix C.</param>
    /// <param name="k">Number of columns in matrix A and rows of matrix B.</param>
    /// <param name="alpha">Scalar for multiplication.</param>
    /// <param name="a">A matrix data pointer.</param>
    /// <param name="lda">Leading dimension of matrix A.</param>
    /// <param name="b">B matrix data pointer.</param>
    /// <param name="ldb">Leading dimension of matrix B.</param>
    /// <param name="beta">Scalar used for multiplication.</param>
    /// <param name="c">Output matrix data.</param>
    /// <param name="ldc">Leading dimension of tensor C.</param>
    public unsafe void Float32GemmV2(
        CublasTransposeType transA,
        CublasTransposeType transB,
        int m,
        int n,
        int k,
        ref float alpha,
        float* a,
        int lda,
        float* b,
        int ldb,
        ref float beta,
        float* c,
        int ldc)
    {
        CublasNativeMethods.Sgemm(
            _handle,
            transA,
            transB,
            m,
            n,
            k,
            ref alpha,
            a,
            lda,
            b,
            ldb,
            ref beta,
            c,
            ldc);
    }

    /// <summary>
    /// Performs a single precision matrix-matrix multiplication.
    /// </summary>
    /// <param name="transA">A operand operation mode.</param>
    /// <param name="transB">B operand operation mode.</param>
    /// <param name="m">Number of rows of matrix A and matrix C.</param>
    /// <param name="n">Number of columns in matrix B and matrix C.</param>
    /// <param name="k">Number of columns in matrix A and rows of matrix B.</param>
    /// <param name="alpha">Scalar for multiplication.</param>
    /// <param name="a">A matrix data pointer.</param>
    /// <param name="lda">Leading dimension of matrix A.</param>
    /// <param name="b">B matrix data pointer.</param>
    /// <param name="ldb">Leading dimension of matrix B.</param>
    /// <param name="beta">Scalar used for multiplication.</param>
    /// <param name="c">Output matrix data.</param>
    /// <param name="ldc">Leading dimension of tensor C.</param>
    public void Float64GemmV2(
        CublasTransposeType transA,
        CublasTransposeType transB,
        int m,
        int n,
        int k,
        ref double alpha,
        TypedMemoryHandle<double> a,
        int lda,
        TypedMemoryHandle<double> b,
        int ldb,
        ref double beta,
        TypedMemoryHandle<double> c,
        int ldc)
    {
        CublasNativeMethods.DgemmV2(
            _handle,
            transA,
            transB,
            m,
            n,
            k,
            ref alpha,
            a.ToUIntPtr(),
            lda,
            b.ToUIntPtr(),
            ldb,
            ref beta,
            c.ToUIntPtr(),
            ldc);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        ReleaseUnmanagedResources();
        GC.SuppressFinalize(this);
    }

    private void ReleaseUnmanagedResources()
    {
        CublasNativeMethods.Destroy(_handle);
    }
#pragma warning restore CA1045
}