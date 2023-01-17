// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// The requested device function does not exist or is not compiled for the proper device architecture.
/// </summary>
[PublicAPI]
public class CudaInvalidDeviceFunctionException : CudaException
{
    private const string DefaultMessage =
        "The requested device function does not exist or is not compiled for the proper device architecture.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidDeviceFunctionException"/> class.
    /// </summary>
    /// <remarks>
    /// The requested device function does not exist or is not compiled for the proper device architecture.
    /// </remarks>
    public CudaInvalidDeviceFunctionException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidDeviceFunctionException"/> class.
    /// </summary>
    /// <remarks>
    /// The requested device function does not exist or is not compiled for the proper device architecture.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidDeviceFunctionException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
