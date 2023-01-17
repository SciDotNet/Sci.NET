// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.BLAS.Layout;

/// <summary>
/// Enumerates the different transpose types.
/// </summary>
[PublicAPI]
public enum TransposeType
{
    /// <summary>
    /// No transpose.
    /// </summary>
    None = 0,

    /// <summary>
    /// Transpose.
    /// </summary>
    Transpose = 1,

    /// <summary>
    /// Conjugate transpose.
    /// </summary>
    ConjugateTranspose = 2
}