// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that a resource is not mapped.
/// </summary>
[PublicAPI]
public class CudaNotMappedException : CudaException
{
    private const string DefaultMessage = "This indicates that a resource is not mapped.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaNotMappedException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a resource is not mapped.
    /// </remarks>
    public CudaNotMappedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaNotMappedException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a resource is not mapped.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaNotMappedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
