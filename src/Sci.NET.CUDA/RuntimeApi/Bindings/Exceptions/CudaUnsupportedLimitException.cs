// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the cudaLimit passed to the API call is not supported by the active device.
/// </summary>
[PublicAPI]
public class CudaUnsupportedLimitException : CudaException
{
    private const string DefaultMessage =
        "This indicates that the cudaLimit passed to the API call is not supported by the active device.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaUnsupportedLimitException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the cudaLimit passed to the API call is not supported by the active device.
    /// </remarks>
    public CudaUnsupportedLimitException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaUnsupportedLimitException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the cudaLimit passed to the API call is not supported by the active device.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaUnsupportedLimitException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
