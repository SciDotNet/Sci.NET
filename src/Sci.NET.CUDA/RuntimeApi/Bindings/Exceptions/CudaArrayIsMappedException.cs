// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the specified array is currently mapped and thus cannot be destroyed.
/// </summary>
[PublicAPI]
public class CudaArrayIsMappedException : CudaException
{
    private const string DefaultMessage =
        "This indicates that the specified array is currently mapped and thus cannot be destroyed.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaArrayIsMappedException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the specified array is currently mapped and thus cannot be destroyed.
    /// </remarks>
    public CudaArrayIsMappedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaArrayIsMappedException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the specified array is currently mapped and thus cannot be destroyed.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaArrayIsMappedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
