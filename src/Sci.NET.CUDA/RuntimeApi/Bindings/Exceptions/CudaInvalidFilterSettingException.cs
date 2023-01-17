// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA.
/// </summary>
[PublicAPI]
public class CudaInvalidFilterSettingException : CudaException
{
    private const string DefaultMessage =
        "This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidFilterSettingException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA.
    /// </remarks>
    public CudaInvalidFilterSettingException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidFilterSettingException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidFilterSettingException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
