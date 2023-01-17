// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that a resource has already been acquired.
/// </summary>
[PublicAPI]
public class CudaAlreadyAcquiredException : CudaException
{
    private const string DefaultMessage = "This indicates that a resource has already been acquired.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaAlreadyAcquiredException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a resource has already been acquired.
    /// </remarks>
    public CudaAlreadyAcquiredException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaAlreadyAcquiredException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a resource has already been acquired.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaAlreadyAcquiredException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
