// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.CuBLAS.Exceptions;

/// <summary>
/// An exception that is thrown because cuBLAS was not initialized.
/// </summary>
[PublicAPI]
public class CublasNotInitializedException : Exception
{
    private const string DefaultMessage =
        "The cuBLAS library was not initialized. This is usually caused by the lack of a prior " +
        "call to CublasCreate(), an error in the CUDA Runtime API called by the cuBLAS routine, or an error in the hardware setup.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CublasNotInitializedException"/> class.
    /// </summary>
    public CublasNotInitializedException()
        : base(DefaultMessage)
    {
    }
}