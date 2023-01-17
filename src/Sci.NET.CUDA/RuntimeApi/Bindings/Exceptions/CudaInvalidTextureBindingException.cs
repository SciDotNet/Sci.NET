// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the texture binding is not valid. This occurs if you call cudaGetTextureAlignmentOffset() with an unbound texture.
/// </summary>
[PublicAPI]
public class CudaInvalidTextureBindingException : CudaException
{
    private const string DefaultMessage =
        "This indicates that the texture binding is not valid. This occurs if you call cudaGetTextureAlignmentOffset() with an unbound texture.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidTextureBindingException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the texture binding is not valid. This occurs if you call cudaGetTextureAlignmentOffset() with an unbound texture.
    /// </remarks>
    public CudaInvalidTextureBindingException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidTextureBindingException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the texture binding is not valid. This occurs if you call cudaGetTextureAlignmentOffset() with an unbound texture.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidTextureBindingException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
