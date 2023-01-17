// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library. This is not a supported configuration. Users should install an updated NVIDIA display driver to allow the application to run.
/// </summary>
[PublicAPI]
public class CudaInsufficientDriverException : CudaException
{
    private const string DefaultMessage =
        "This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library. This is not a supported configuration. Users should install an updated NVIDIA display driver to allow the application to run.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInsufficientDriverException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library. This is not a supported configuration. Users should install an updated NVIDIA display driver to allow the application to run.
    /// </remarks>
    public CudaInsufficientDriverException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInsufficientDriverException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library. This is not a supported configuration. Users should install an updated NVIDIA display driver to allow the application to run.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInsufficientDriverException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
