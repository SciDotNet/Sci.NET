// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// The current capture sequence on the stream has been invalidated due to a previous error.
/// </summary>
[PublicAPI]
public class CudaStreamCaptureInvalidatedException : CudaException
{
    private const string DefaultMessage =
        "The current capture sequence on the stream has been invalidated due to a previous error.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStreamCaptureInvalidatedException"/> class.
    /// </summary>
    /// <remarks>
    /// The current capture sequence on the stream has been invalidated due to a previous error.
    /// </remarks>
    public CudaStreamCaptureInvalidatedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStreamCaptureInvalidatedException"/> class.
    /// </summary>
    /// <remarks>
    /// The current capture sequence on the stream has been invalidated due to a previous error.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaStreamCaptureInvalidatedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
