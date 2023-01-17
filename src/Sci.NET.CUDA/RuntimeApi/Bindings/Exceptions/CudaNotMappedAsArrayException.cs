// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that a mapped resource is not available for access as an array.
/// </summary>
[PublicAPI]
public class CudaNotMappedAsArrayException : CudaException
{
    private const string DefaultMessage =
        "This indicates that a mapped resource is not available for access as an array.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaNotMappedAsArrayException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a mapped resource is not available for access as an array.
    /// </remarks>
    public CudaNotMappedAsArrayException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaNotMappedAsArrayException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a mapped resource is not available for access as an array.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaNotMappedAsArrayException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
