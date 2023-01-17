// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the API call requires a newer CUDA driver than the one currently installed. Users should install an updated NVIDIA CUDA driver to allow the API call to succeed.
/// </summary>
[PublicAPI]
public class CudaCallRequiresNewerDriverException : CudaException
{
    private const string DefaultMessage =
        "This indicates that the API call requires a newer CUDA driver than the one currently installed. Users should install an updated NVIDIA CUDA driver to allow the API call to succeed.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaCallRequiresNewerDriverException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the API call requires a newer CUDA driver than the one currently installed. Users should install an updated NVIDIA CUDA driver to allow the API call to succeed.
    /// </remarks>
    public CudaCallRequiresNewerDriverException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaCallRequiresNewerDriverException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the API call requires a newer CUDA driver than the one currently installed. Users should install an updated NVIDIA CUDA driver to allow the API call to succeed.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaCallRequiresNewerDriverException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
