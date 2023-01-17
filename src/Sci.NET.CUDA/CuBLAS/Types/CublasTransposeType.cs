// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.CuBLAS.Types;

/// <summary>
/// Specifies the type of operation to be performed.
/// </summary>
[PublicAPI]
#pragma warning disable CA1027
public enum CublasTransposeType
#pragma warning restore CA1027
{
    /// <summary>
    /// The operation is non-transpose.
    /// </summary>
    CublasOpN = 0,

    /// <summary>
    /// The operation is transpose.
    /// </summary>
    CublasOpT = 1,

    /// <summary>
    /// The operation is conjugate transpose.
    /// </summary>
    CublasOpC = 2,

    /// <summary>
    /// The operation is conjugate transpose (synonym of <see cref="CublasOpC"/>).
    /// </summary>
    CublasOpHermitan = CublasOpC,

    /// <summary>
    /// The operation is conjugate (Not yet supported).
    /// </summary>
    CublasOpConj = 3
}