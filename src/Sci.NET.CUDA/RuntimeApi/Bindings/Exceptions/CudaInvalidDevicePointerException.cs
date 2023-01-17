// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that at least one device pointer passed to the API call is not a valid device pointer.
/// </summary>
[PublicAPI]
public class CudaInvalidDevicePointerException : CudaException
{
    private const string DefaultMessage =
        "This indicates that at least one device pointer passed to the API call is not a valid device pointer.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidDevicePointerException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that at least one device pointer passed to the API call is not a valid device pointer.
    /// </remarks>
    public CudaInvalidDevicePointerException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidDevicePointerException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that at least one device pointer passed to the API call is not a valid device pointer.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidDevicePointerException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
