// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.CuBLAS.Exceptions;

/// <summary>
/// An exception thrown when a cuBLAS operation fails.
/// </summary>
[PublicAPI]
public class CublasExecutionFailedException : Exception
{
    private const string DefaultMessage =
        "The cuBLAS operation failed.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CublasExecutionFailedException"/> class.
    /// </summary>
    public CublasExecutionFailedException()
        : base(DefaultMessage)
    {
    }
}