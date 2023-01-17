// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// A stream capture sequence not initiated with the cudaStreamCaptureModeRelaxed argument to cudaStreamBeginCapture was passed to cudaStreamEndCapture in a different thread.
/// </summary>
[PublicAPI]
public class CudaStreamCaptureWrongThreadException : CudaException
{
    private const string DefaultMessage =
        "A stream capture sequence not initiated with the cudaStreamCaptureModeRelaxed argument to cudaStreamBeginCapture was passed to cudaStreamEndCapture in a different thread.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStreamCaptureWrongThreadException"/> class.
    /// </summary>
    /// <remarks>
    /// A stream capture sequence not initiated with the cudaStreamCaptureModeRelaxed argument to cudaStreamBeginCapture was passed to cudaStreamEndCapture in a different thread.
    /// </remarks>
    public CudaStreamCaptureWrongThreadException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStreamCaptureWrongThreadException"/> class.
    /// </summary>
    /// <remarks>
    /// A stream capture sequence not initiated with the cudaStreamCaptureModeRelaxed argument to cudaStreamBeginCapture was passed to cudaStreamEndCapture in a different thread.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaStreamCaptureWrongThreadException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
