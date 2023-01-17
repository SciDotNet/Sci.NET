// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device or that the action requested is invalid for the specified device.
/// </summary>
[PublicAPI]
public class CudaInvalidDeviceException : CudaException
{
    private const string DefaultMessage =
        "This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device or that the action requested is invalid for the specified device.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidDeviceException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device or that the action requested is invalid for the specified device.
    /// </remarks>
    public CudaInvalidDeviceException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidDeviceException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device or that the action requested is invalid for the specified device.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidDeviceException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
