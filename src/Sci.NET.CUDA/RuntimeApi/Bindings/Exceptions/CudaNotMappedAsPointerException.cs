// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that a mapped resource is not available for access as a pointer.
/// </summary>
[PublicAPI]
public class CudaNotMappedAsPointerException : CudaException
{
    private const string DefaultMessage =
        "This indicates that a mapped resource is not available for access as a pointer.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaNotMappedAsPointerException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a mapped resource is not available for access as a pointer.
    /// </remarks>
    public CudaNotMappedAsPointerException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaNotMappedAsPointerException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a mapped resource is not available for access as a pointer.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaNotMappedAsPointerException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
