// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.CuBLAS.Exceptions;

/// <summary>
/// An exception thrown when the cuBLAS API fails internally.
/// </summary>
[PublicAPI]
public class CublasInternalErrorException : Exception
{
    private const string DefaultMessage =
        "An internal operation failed. This error is usually caused by a cudaMemcpyAsync() failure.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CublasInternalErrorException"/> class.
    /// </summary>
    public CublasInternalErrorException()
        : base(DefaultMessage)
    {
    }
}