// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates that P2P access is not supported across the given devices.
/// </summary>
[PublicAPI]
public class CudaPeerAccessUnsupportedException : CudaException
{
    private const string DefaultMessage =
        "This error indicates that P2P access is not supported across the given devices.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaPeerAccessUnsupportedException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that P2P access is not supported across the given devices.
    /// </remarks>
    public CudaPeerAccessUnsupportedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaPeerAccessUnsupportedException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that P2P access is not supported across the given devices.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaPeerAccessUnsupportedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
