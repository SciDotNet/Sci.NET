// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using Sci.NET.CUDA.CuBLAS.Extensions;
using Sci.NET.CUDA.CuBLAS.Types;

namespace Sci.NET.CUDA.CuBLAS;

#pragma warning disable CA1060, CA5392
internal static class CublasNativeMethods
{
    private const string CublasLibraryName = "cublas64_12";

    public static OpaqueCublasHandle Create()
    {
        var handle = default(OpaqueCublasHandle);
        CublasCreateV2(ref handle)
            .Guard();
        return handle;
    }

    public static void Destroy(OpaqueCublasHandle handle)
    {
        CublasDestroyV2(handle)
            .Guard();
    }

    public static unsafe void Sgemm(
        OpaqueCublasHandle handle,
        CublasTransposeType transA,
        CublasTransposeType transB,
        int m,
        int n,
        int k,
        ref float alpha,
        float* A,
        int lda,
        float* B,
        int ldb,
        ref float beta,
        float* C,
        int ldc)
    {
        CublasSgemm(
                handle,
                transA,
                transB,
                m,
                n,
                k,
                ref alpha,
                A,
                lda,
                B,
                ldb,
                ref beta,
                C,
                ldc)
            .Guard();
    }

    public static void DgemmV2(
        OpaqueCublasHandle handle,
        CublasTransposeType transA,
        CublasTransposeType transB,
        int m,
        int n,
        int k,
        ref double alpha,
        nuint A,
        int lda,
        nuint B,
        int ldb,
        ref double beta,
        nuint C,
        int ldc)
    {
        CublasDgemmV2(
                handle,
                transA,
                transB,
                m,
                n,
                k,
                ref alpha,
                A,
                lda,
                B,
                ldb,
                ref beta,
                C,
                ldc)
            .Guard();
    }

    [DllImport(CublasLibraryName, EntryPoint = "cublasCreate_v2")]
    private static extern CublasStatus CublasCreateV2(ref OpaqueCublasHandle handle);

    [DllImport(CublasLibraryName, EntryPoint = "cublasDestroy_v2")]
    private static extern CublasStatus CublasDestroyV2(OpaqueCublasHandle handle);

    [DllImport(CublasLibraryName, EntryPoint = "cublasSgemm")]
    private static extern unsafe CublasStatus CublasSgemm(
        OpaqueCublasHandle handle,
        CublasTransposeType transA,
        CublasTransposeType transB,
        int m,
        int n,
        int k,
        ref float alpha,
        float* A,
        int lda,
        float* B,
        int ldb,
        ref float beta,
        float* C,
        int ldc);

    [DllImport(CublasLibraryName, EntryPoint = "cublasSgemm_v2")]
    private static extern CublasStatus CublasDgemmV2(
        OpaqueCublasHandle handle,
        CublasTransposeType transA,
        CublasTransposeType transB,
        int m,
        int n,
        int k,
        ref double alpha,
        nuint A,
        int lda,
        nuint B,
        int ldb,
        ref double beta,
        nuint C,
        int ldc);
}
#pragma warning restore CA1060, CA5392