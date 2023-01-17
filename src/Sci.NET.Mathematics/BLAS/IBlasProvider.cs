// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.BLAS.Layout;

namespace Sci.NET.Mathematics.BLAS;

/// <summary>
/// A clas that implements BLAS routines.
/// </summary>
public interface IBlasProvider
{
     /// <summary>
    /// Performs a matrix-matrix multiplication using row-major matrices
    /// on the values <paramref name="a"/> and <paramref name="b"/>.
    /// </summary>
    /// <param name="transA">The transpose operation to be performed on matrix <paramref name="a"/>.</param>
    /// <param name="transB">The transpose operation to be performed on matrix <paramref name="b"/>.</param>
    /// <param name="leftDimX">The number of rows of matrix <paramref name="c"/>, which is the output matrix.</param>
    /// <param name="rightDimY">The number of columns in matrix <paramref name="c"/>, which is the output matrix.</param>
    /// <param name="leftDimY">
    /// The common dimensions of <paramref name="a"/> and <paramref name="b"/>, which is the number of columns
    /// in matrix <paramref name="a"/> and the number of rows in matrix <paramref name="b"/> when no transpose
    /// is performed, or the number of rows in matrix <paramref name="a"/> and the number of columns in matrix
    /// <paramref name="b"/> when a transpose is performed.
    /// </param>
    /// <param name="alpha">A scalar value that is multiplied by the result of the matrix-matrix multiplication.</param>
    /// <param name="a">A data array containing the elements of matrix <paramref name="a"/>.</param>
    /// <param name="lda">The leading dimension of matrix <paramref name="a"/>.</param>
    /// <param name="b">A data array containing the elements of matrix <paramref name="b"/>.</param>
    /// <param name="ldb">The leading dimension of matrix <paramref name="b"/>.</param>
    /// <param name="beta">a scalar value that is multiplied by the elements of matrix <paramref name="c"/>
    /// before adding the result of the matrix-matrix multiplication.</param>
    /// <param name="c">A data array containing the elements of matrix <paramref name="c"/>.</param>
    /// <param name="ldc">The leading dimension of matrix <paramref name="c"/>.</param>
    /// <typeparam name="T">The number type of the matrix.</typeparam>
    /// <exception cref="ArgumentException">Throws when an invalid value is passed.</exception>
    public void Gemm<T>(
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
        where T : unmanaged, INumber<T>;

    /// <summary>
    /// Sets the maximum number of threads that can be used by the BLAS library.
    /// </summary>
    /// <param name="maxThreads">The maximum number of threads to be used.</param>
    public void SetMaxThreads(int maxThreads);
}