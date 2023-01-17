// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the device doesn't have a valid Grid License.
/// </summary>
[PublicAPI]
public class CudaDeviceNotLicensedException : CudaException
{
    private const string DefaultMessage = "This indicates that the device doesn't have a valid Grid License.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaDeviceNotLicensedException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the device doesn't have a valid Grid License.
    /// </remarks>
    public CudaDeviceNotLicensedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaDeviceNotLicensedException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the device doesn't have a valid Grid License.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaDeviceNotLicensedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
