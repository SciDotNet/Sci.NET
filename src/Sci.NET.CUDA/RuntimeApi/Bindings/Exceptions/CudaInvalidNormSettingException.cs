// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that an attempt was made to read a non-float texture as a normalized float. This is not supported by CUDA.
/// </summary>
[PublicAPI]
public class CudaInvalidNormSettingException : CudaException
{
    private const string DefaultMessage =
        "This indicates that an attempt was made to read a non-float texture as a normalized float. This is not supported by CUDA.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidNormSettingException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that an attempt was made to read a non-float texture as a normalized float. This is not supported by CUDA.
    /// </remarks>
    public CudaInvalidNormSettingException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidNormSettingException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that an attempt was made to read a non-float texture as a normalized float. This is not supported by CUDA.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidNormSettingException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
