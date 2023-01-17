// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates that the remote procedural call between the MPS server and the MPS client failed.
/// </summary>
[PublicAPI]
public class CudaMpsRpcFailureException : CudaException
{
    private const string DefaultMessage =
        "This error indicates that the remote procedural call between the MPS server and the MPS client failed.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMpsRpcFailureException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the remote procedural call between the MPS server and the MPS client failed.
    /// </remarks>
    public CudaMpsRpcFailureException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMpsRpcFailureException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the remote procedural call between the MPS server and the MPS client failed.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaMpsRpcFailureException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
