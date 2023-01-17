// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the direction of the memcpy passed to the API call is not one of the types specified by cudaMemcpyKind.
/// </summary>
[PublicAPI]
public class CudaInvalidMemcpyDirectionException : CudaException
{
    private const string DefaultMessage =
        "This indicates that the direction of the memcpy passed to the API call is not one of the types specified by cudaMemcpyKind.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidMemcpyDirectionException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the direction of the memcpy passed to the API call is not one of the types specified by cudaMemcpyKind.
    /// </remarks>
    public CudaInvalidMemcpyDirectionException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidMemcpyDirectionException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the direction of the memcpy passed to the API call is not one of the types specified by cudaMemcpyKind.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidMemcpyDirectionException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
