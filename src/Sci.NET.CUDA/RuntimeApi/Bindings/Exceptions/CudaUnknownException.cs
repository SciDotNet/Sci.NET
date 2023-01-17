// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that an unknown internal error has occurred.
/// </summary>
[PublicAPI]
public class CudaUnknownException : CudaException
{
    private const string DefaultMessage = "This indicates that an unknown internal error has occurred.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaUnknownException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that an unknown internal error has occurred.
    /// </remarks>
    public CudaUnknownException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaUnknownException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that an unknown internal error has occurred.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaUnknownException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
