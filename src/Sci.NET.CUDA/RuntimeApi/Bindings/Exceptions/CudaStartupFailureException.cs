// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates an internal startup failure in the CUDA runtime.
/// </summary>
[PublicAPI]
public class CudaStartupFailureException : CudaException
{
    private const string DefaultMessage = "This indicates an internal startup failure in the CUDA runtime.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStartupFailureException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates an internal startup failure in the CUDA runtime.
    /// </remarks>
    public CudaStartupFailureException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStartupFailureException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates an internal startup failure in the CUDA runtime.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaStartupFailureException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
