// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates an error with the OpenGL or DirectX context.
/// </summary>
[PublicAPI]
public class CudaInvalidGraphicsContextException : CudaException
{
    private const string DefaultMessage = "This indicates an error with the OpenGL or DirectX context.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidGraphicsContextException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates an error with the OpenGL or DirectX context.
    /// </remarks>
    public CudaInvalidGraphicsContextException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidGraphicsContextException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates an error with the OpenGL or DirectX context.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidGraphicsContextException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
