// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that a call tried to access an exclusive-thread device that is already in use by a different thread.
/// </summary>
[PublicAPI]
public class CudaDeviceAlreadyInUseException : CudaException
{
    private const string DefaultMessage =
        "This indicates that a call tried to access an exclusive-thread device that is already in use by a different thread.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaDeviceAlreadyInUseException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a call tried to access an exclusive-thread device that is already in use by a different thread.
    /// </remarks>
    public CudaDeviceAlreadyInUseException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaDeviceAlreadyInUseException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a call tried to access an exclusive-thread device that is already in use by a different thread.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaDeviceAlreadyInUseException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
