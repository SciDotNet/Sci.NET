// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// A PTX compilation failed. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.
/// </summary>
[PublicAPI]
public class CudaInvalidPtxException : CudaException
{
    private const string DefaultMessage =
        "A PTX compilation failed. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidPtxException"/> class.
    /// </summary>
    /// <remarks>
    /// A PTX compilation failed. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.
    /// </remarks>
    public CudaInvalidPtxException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidPtxException"/> class.
    /// </summary>
    /// <remarks>
    /// A PTX compilation failed. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidPtxException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
