// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions.
/// </summary>
[PublicAPI]
public class CudaSystemDriverMismatchException : CudaException
{
    private const string DefaultMessage =
        "This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaSystemDriverMismatchException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions.
    /// </remarks>
    public CudaSystemDriverMismatchException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaSystemDriverMismatchException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaSystemDriverMismatchException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
