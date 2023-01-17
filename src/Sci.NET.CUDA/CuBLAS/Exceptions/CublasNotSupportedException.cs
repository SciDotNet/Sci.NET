// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.CuBLAS.Exceptions;

/// <summary>
/// An exception thrown when the requested functionality
/// is not supported by the cuBLAS library.
/// </summary>
[PublicAPI]
public class CublasNotSupportedException : Exception
{
    private const string DefaultMessage =
        "The requested functionailty is not supported by the cuBLAS library.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CublasNotSupportedException"/> class.
    /// </summary>
    public CublasNotSupportedException()
        : base(DefaultMessage)
    {
    }
}