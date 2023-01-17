// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates the hardware resources required to device connections have been exhausted.
/// </summary>
[PublicAPI]
public class CudaMpsMaxConnectionsReachedException : CudaException
{
    private const string DefaultMessage =
        "This error indicates the the hardware resources required to device connections have been exhausted.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMpsMaxConnectionsReachedException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates the hardware resources required to device connections have been exhausted.
    /// </remarks>
    public CudaMpsMaxConnectionsReachedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMpsMaxConnectionsReachedException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates the hardware resources required to device connections have been exhausted.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaMpsMaxConnectionsReachedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
