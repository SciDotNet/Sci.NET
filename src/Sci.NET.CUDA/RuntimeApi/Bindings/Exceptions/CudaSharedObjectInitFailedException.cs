// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that initialization of a shared object failed.
/// </summary>
[PublicAPI]
public class CudaSharedObjectInitFailedException : CudaException
{
    private const string DefaultMessage = "This indicates that initialization of a shared object failed.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaSharedObjectInitFailedException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that initialization of a shared object failed.
    /// </remarks>
    public CudaSharedObjectInitFailedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaSharedObjectInitFailedException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that initialization of a shared object failed.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaSharedObjectInitFailedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
