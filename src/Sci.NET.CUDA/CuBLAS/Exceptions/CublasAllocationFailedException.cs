// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.CuBLAS.Exceptions;

/// <summary>
/// The exception that is thrown when a cuBLAS allocation failed.
/// </summary>
[PublicAPI]
public class CublasAllocationFailedException : Exception
{
    private const string DefaultMessage =
        "Resource allocation failed inside the cuBLAS library. This is usually caused by a cudaMalloc() failure.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CublasAllocationFailedException"/> class.
    /// </summary>
    public CublasAllocationFailedException()
        : base(DefaultMessage)
    {
    }
}