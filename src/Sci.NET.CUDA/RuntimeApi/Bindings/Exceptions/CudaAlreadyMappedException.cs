// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the resource is already mapped.
/// </summary>
[PublicAPI]
public class CudaAlreadyMappedException : CudaException
{
    private const string DefaultMessage = "This indicates that the resource is already mapped.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaAlreadyMappedException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the resource is already mapped.
    /// </remarks>
    public CudaAlreadyMappedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaAlreadyMappedException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the resource is already mapped.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaAlreadyMappedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
