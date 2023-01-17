// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.
/// </summary>
[PublicAPI]
public class CudaCompatNotSupportedOnDeviceException : CudaException
{
    private const string DefaultMessage =
        "This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaCompatNotSupportedOnDeviceException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.
    /// </remarks>
    public CudaCompatNotSupportedOnDeviceException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaCompatNotSupportedOnDeviceException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaCompatNotSupportedOnDeviceException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
