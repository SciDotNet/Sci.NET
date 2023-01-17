// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the surface passed to the API call is not a valid surface.
/// </summary>
[PublicAPI]
public class CudaInvalidSurfaceException : CudaException
{
    private const string DefaultMessage =
        "This indicates that the surface passed to the API call is not a valid surface.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidSurfaceException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the surface passed to the API call is not a valid surface.
    /// </remarks>
    public CudaInvalidSurfaceException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidSurfaceException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the surface passed to the API call is not a valid surface.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidSurfaceException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
