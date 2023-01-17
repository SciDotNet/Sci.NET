// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that no CUDA-capable devices were detected by the installed CUDA driver.
/// </summary>
[PublicAPI]
public class CudaNoDeviceException : CudaException
{
    private const string DefaultMessage =
        "This indicates that no CUDA-capable devices were detected by the installed CUDA driver.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaNoDeviceException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that no CUDA-capable devices were detected by the installed CUDA driver.
    /// </remarks>
    public CudaNoDeviceException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaNoDeviceException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that no CUDA-capable devices were detected by the installed CUDA driver.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaNoDeviceException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
