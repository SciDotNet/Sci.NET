// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// The capture sequence contains a fork that was not joined to the primary stream.
/// </summary>
[PublicAPI]
public class CudaStreamCaptureUnjoinedException : CudaException
{
    private const string DefaultMessage =
        "The capture sequence contains a fork that was not joined to the primary stream.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStreamCaptureUnjoinedException"/> class.
    /// </summary>
    /// <remarks>
    /// The capture sequence contains a fork that was not joined to the primary stream.
    /// </remarks>
    public CudaStreamCaptureUnjoinedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStreamCaptureUnjoinedException"/> class.
    /// </summary>
    /// <remarks>
    /// The capture sequence contains a fork that was not joined to the primary stream.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaStreamCaptureUnjoinedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
