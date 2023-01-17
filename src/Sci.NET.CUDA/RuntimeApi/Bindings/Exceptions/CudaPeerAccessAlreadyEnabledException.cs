// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates that a call to cudaDeviceEnablePeerAccess() is trying to re-enable peer addressing on from a context which has already had peer addressing enabled.
/// </summary>
[PublicAPI]
public class CudaPeerAccessAlreadyEnabledException : CudaException
{
    private const string DefaultMessage =
        "This error indicates that a call to cudaDeviceEnablePeerAccess() is trying to re-enable peer addressing on from a context which has already had peer addressing enabled.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaPeerAccessAlreadyEnabledException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that a call to cudaDeviceEnablePeerAccess() is trying to re-enable peer addressing on from a context which has already had peer addressing enabled.
    /// </remarks>
    public CudaPeerAccessAlreadyEnabledException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaPeerAccessAlreadyEnabledException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that a call to cudaDeviceEnablePeerAccess() is trying to re-enable peer addressing on from a context which has already had peer addressing enabled.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaPeerAccessAlreadyEnabledException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
