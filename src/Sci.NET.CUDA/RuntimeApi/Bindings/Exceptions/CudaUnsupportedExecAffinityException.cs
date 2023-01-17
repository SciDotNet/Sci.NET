// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the provided execution affinity is not supported by the device.
/// </summary>
[PublicAPI]
public class CudaUnsupportedExecAffinityException : CudaException
{
    private const string DefaultMessage =
        "This indicates that the provided execution affinity is not supported by the device.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaUnsupportedExecAffinityException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the provided execution affinity is not supported by the device.
    /// </remarks>
    public CudaUnsupportedExecAffinityException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaUnsupportedExecAffinityException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the provided execution affinity is not supported by the device.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaUnsupportedExecAffinityException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
