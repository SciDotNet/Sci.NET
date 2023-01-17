// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that a resource required by the API call is not in a valid state to perform the requested operation.
/// </summary>
[PublicAPI]
public class CudaIllegalStateException : CudaException
{
    private const string DefaultMessage =
        "This indicates that a resource required by the API call is not in a valid state to perform the requested operation.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaIllegalStateException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a resource required by the API call is not in a valid state to perform the requested operation.
    /// </remarks>
    public CudaIllegalStateException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaIllegalStateException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a resource required by the API call is not in a valid state to perform the requested operation.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaIllegalStateException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
