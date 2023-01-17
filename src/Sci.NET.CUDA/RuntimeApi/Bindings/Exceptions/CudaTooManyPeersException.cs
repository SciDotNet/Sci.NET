// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to cudaEnablePeerAccess().
/// </summary>
[PublicAPI]
public class CudaTooManyPeersException : CudaException
{
    private const string DefaultMessage =
        "This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to cudaEnablePeerAccess().";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaTooManyPeersException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to cudaEnablePeerAccess().
    /// </remarks>
    public CudaTooManyPeersException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaTooManyPeersException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to cudaEnablePeerAccess().
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaTooManyPeersException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
