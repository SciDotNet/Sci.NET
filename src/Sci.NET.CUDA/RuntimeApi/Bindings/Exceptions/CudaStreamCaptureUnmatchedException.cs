// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// The capture was not initiated in this stream.
/// </summary>
[PublicAPI]
public class CudaStreamCaptureUnmatchedException : CudaException
{
    private const string DefaultMessage = "The capture was not initiated in this stream.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStreamCaptureUnmatchedException"/> class.
    /// </summary>
    /// <remarks>
    /// The capture was not initiated in this stream.
    /// </remarks>
    public CudaStreamCaptureUnmatchedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStreamCaptureUnmatchedException"/> class.
    /// </summary>
    /// <remarks>
    /// The capture was not initiated in this stream.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaStreamCaptureUnmatchedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
