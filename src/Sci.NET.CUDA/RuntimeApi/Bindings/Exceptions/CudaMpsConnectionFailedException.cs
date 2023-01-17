// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.
/// </summary>
[PublicAPI]
public class CudaMpsConnectionFailedException : CudaException
{
    private const string DefaultMessage =
        "This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMpsConnectionFailedException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.
    /// </remarks>
    public CudaMpsConnectionFailedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMpsConnectionFailedException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaMpsConnectionFailedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
