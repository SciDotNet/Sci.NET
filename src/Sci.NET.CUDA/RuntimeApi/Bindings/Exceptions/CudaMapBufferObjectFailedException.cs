// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the buffer object could not be mapped.
/// </summary>
[PublicAPI]
public class CudaMapBufferObjectFailedException : CudaException
{
    private const string DefaultMessage = "This indicates that the buffer object could not be mapped.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMapBufferObjectFailedException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the buffer object could not be mapped.
    /// </remarks>
    public CudaMapBufferObjectFailedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMapBufferObjectFailedException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the buffer object could not be mapped.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaMapBufferObjectFailedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
