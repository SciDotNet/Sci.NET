// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates that cudaDeviceDisablePeerAccess() is trying to disable peer addressing which has not been enabled yet via cudaDeviceEnablePeerAccess().
/// </summary>
[PublicAPI]
public class CudaPeerAccessNotEnabledException : CudaException
{
    private const string DefaultMessage =
        "This error indicates that cudaDeviceDisablePeerAccess() is trying to disable peer addressing which has not been enabled yet via cudaDeviceEnablePeerAccess().";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaPeerAccessNotEnabledException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that cudaDeviceDisablePeerAccess() is trying to disable peer addressing which has not been enabled yet via cudaDeviceEnablePeerAccess().
    /// </remarks>
    public CudaPeerAccessNotEnabledException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaPeerAccessNotEnabledException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that cudaDeviceDisablePeerAccess() is trying to disable peer addressing which has not been enabled yet via cudaDeviceEnablePeerAccess().
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaPeerAccessNotEnabledException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
