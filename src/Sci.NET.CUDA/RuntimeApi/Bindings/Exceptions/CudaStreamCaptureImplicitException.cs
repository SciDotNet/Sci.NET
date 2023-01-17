// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// The operation would have resulted in a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.
/// </summary>
[PublicAPI]
public class CudaStreamCaptureImplicitException : CudaException
{
    private const string DefaultMessage =
        "The operation would have resulted in a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStreamCaptureImplicitException"/> class.
    /// </summary>
    /// <remarks>
    /// The operation would have resulted in a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.
    /// </remarks>
    public CudaStreamCaptureImplicitException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStreamCaptureImplicitException"/> class.
    /// </summary>
    /// <remarks>
    /// The operation would have resulted in a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaStreamCaptureImplicitException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
