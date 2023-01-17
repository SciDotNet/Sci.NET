// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates that the hardware resources required to create MPS client have been exhausted.
/// </summary>
[PublicAPI]
public class CudaMpsMaxClientsReachedException : CudaException
{
    private const string DefaultMessage =
        "This error indicates that the hardware resources required to create MPS client have been exhausted.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMpsMaxClientsReachedException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the hardware resources required to create MPS client have been exhausted.
    /// </remarks>
    public CudaMpsMaxClientsReachedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMpsMaxClientsReachedException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the hardware resources required to create MPS client have been exhausted.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaMpsMaxClientsReachedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
