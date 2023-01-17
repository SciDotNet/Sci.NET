// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the buffer object could not be unmapped.
/// </summary>
[PublicAPI]
public class CudaUnmapBufferObjectFailedException : CudaException
{
    private const string DefaultMessage = "This indicates that the buffer object could not be unmapped.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaUnmapBufferObjectFailedException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the buffer object could not be unmapped.
    /// </remarks>
    public CudaUnmapBufferObjectFailedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaUnmapBufferObjectFailedException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the buffer object could not be unmapped.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaUnmapBufferObjectFailedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
