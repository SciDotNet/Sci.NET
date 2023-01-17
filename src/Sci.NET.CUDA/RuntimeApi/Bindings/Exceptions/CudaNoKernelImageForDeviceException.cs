// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.
/// </summary>
[PublicAPI]
public class CudaNoKernelImageForDeviceException : CudaException
{
    private const string DefaultMessage =
        "This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaNoKernelImageForDeviceException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.
    /// </remarks>
    public CudaNoKernelImageForDeviceException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaNoKernelImageForDeviceException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaNoKernelImageForDeviceException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
