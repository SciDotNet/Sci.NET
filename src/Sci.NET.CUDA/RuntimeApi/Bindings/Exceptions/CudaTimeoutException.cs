// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the wait operation has timed out.
/// </summary>
[PublicAPI]
public class CudaTimeoutException : CudaException
{
    private const string DefaultMessage = "This indicates that the wait operation has timed out.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaTimeoutException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the wait operation has timed out.
    /// </remarks>
    public CudaTimeoutException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaTimeoutException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the wait operation has timed out.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaTimeoutException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
