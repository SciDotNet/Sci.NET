// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates the attempted operation is not supported on the current system or device.
/// </summary>
[PublicAPI]
public class CudaNotSupportedException : CudaException
{
    private const string DefaultMessage =
        "This error indicates the attempted operation is not supported on the current system or device.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaNotSupportedException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates the attempted operation is not supported on the current system or device.
    /// </remarks>
    public CudaNotSupportedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaNotSupportedException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates the attempted operation is not supported on the current system or device.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaNotSupportedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
