// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the device kernel source is invalid.
/// </summary>
[PublicAPI]
public class CudaInvalidSourceException : CudaException
{
    private const string DefaultMessage = "This indicates that the device kernel source is invalid.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidSourceException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the device kernel source is invalid.
    /// </remarks>
    public CudaInvalidSourceException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidSourceException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the device kernel source is invalid.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidSourceException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
