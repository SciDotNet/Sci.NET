// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates that the MPS server is not ready to accept new MPS client requests. This error can be returned when the MPS server is in the process of recovering from a fatal failure.
/// </summary>
[PublicAPI]
public class CudaMpsServerNotReadyException : CudaException
{
    private const string DefaultMessage =
        "This error indicates that the MPS server is not ready to accept new MPS client requests. This error can be returned when the MPS server is in the process of recovering from a fatal failure.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMpsServerNotReadyException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the MPS server is not ready to accept new MPS client requests. This error can be returned when the MPS server is in the process of recovering from a fatal failure.
    /// </remarks>
    public CudaMpsServerNotReadyException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMpsServerNotReadyException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the MPS server is not ready to accept new MPS client requests. This error can be returned when the MPS server is in the process of recovering from a fatal failure.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaMpsServerNotReadyException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
