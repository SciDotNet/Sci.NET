// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.CUDA.Runtime.Structs;

namespace Sci.NET.CUDA.Runtime.Exceptions;

/// <summary>
/// An exception thrown when a CUDA runtime API call fails.
/// </summary>
[PublicAPI]
public class CudaRuntimeException : Exception
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CudaRuntimeException"/> class.
    /// </summary>
    /// <param name="statusCode">The status code returned by the runtime API.</param>
    public CudaRuntimeException(CudaStatusCode statusCode)
        : base($"CUDA runtime API call failed with status code {statusCode}.")
    {
        StatusCode = statusCode;
    }

    /// <summary>
    /// Gets the status code returned by the CUDA runtime API call.
    /// </summary>
    public CudaStatusCode StatusCode { get; }
}