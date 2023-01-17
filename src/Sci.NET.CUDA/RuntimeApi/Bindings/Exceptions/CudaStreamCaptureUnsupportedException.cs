// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// The operation is not permitted when the stream is capturing.
/// </summary>
[PublicAPI]
public class CudaStreamCaptureUnsupportedException : CudaException
{
    private const string DefaultMessage = "The operation is not permitted when the stream is capturing.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStreamCaptureUnsupportedException"/> class.
    /// </summary>
    /// <remarks>
    /// The operation is not permitted when the stream is capturing.
    /// </remarks>
    public CudaStreamCaptureUnsupportedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStreamCaptureUnsupportedException"/> class.
    /// </summary>
    /// <remarks>
    /// The operation is not permitted when the stream is capturing.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaStreamCaptureUnsupportedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
