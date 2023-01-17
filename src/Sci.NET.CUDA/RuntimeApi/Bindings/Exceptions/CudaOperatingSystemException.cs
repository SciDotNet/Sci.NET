// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates that an OS call failed.
/// </summary>
[PublicAPI]
public class CudaOperatingSystemException : CudaException
{
    private const string DefaultMessage = "This error indicates that an OS call failed.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaOperatingSystemException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that an OS call failed.
    /// </remarks>
    public CudaOperatingSystemException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaOperatingSystemException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that an OS call failed.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaOperatingSystemException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
