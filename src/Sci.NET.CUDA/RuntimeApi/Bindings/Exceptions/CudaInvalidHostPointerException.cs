// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that at least one host pointer passed to the API call is not a valid host pointer.
/// </summary>
[PublicAPI]
public class CudaInvalidHostPointerException : CudaException
{
    private const string DefaultMessage =
        "This indicates that at least one host pointer passed to the API call is not a valid host pointer.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidHostPointerException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that at least one host pointer passed to the API call is not a valid host pointer.
    /// </remarks>
    public CudaInvalidHostPointerException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidHostPointerException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that at least one host pointer passed to the API call is not a valid host pointer.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidHostPointerException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
