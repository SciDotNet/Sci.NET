// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// A dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.
/// </summary>
[PublicAPI]
public class CudaStreamCaptureIsolationException : CudaException
{
    private const string DefaultMessage =
        "A dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStreamCaptureIsolationException"/> class.
    /// </summary>
    /// <remarks>
    /// A dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.
    /// </remarks>
    public CudaStreamCaptureIsolationException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStreamCaptureIsolationException"/> class.
    /// </summary>
    /// <remarks>
    /// A dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaStreamCaptureIsolationException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
