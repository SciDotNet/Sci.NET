// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// The API call failed because it was unable to allocate enough memory to perform the requested operation.
/// </summary>
[PublicAPI]
public class CudaMemoryAllocationException : CudaException
{
    private const string DefaultMessage =
        "The API call failed because it was unable to allocate enough memory to perform the requested operation.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMemoryAllocationException"/> class.
    /// </summary>
    /// <remarks>
    /// The API call failed because it was unable to allocate enough memory to perform the requested operation.
    /// </remarks>
    public CudaMemoryAllocationException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMemoryAllocationException"/> class.
    /// </summary>
    /// <remarks>
    /// The API call failed because it was unable to sdn_allocate enough memory to perform the requested operation.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaMemoryAllocationException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
