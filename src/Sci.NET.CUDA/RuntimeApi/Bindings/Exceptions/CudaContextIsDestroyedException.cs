// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates that the context current to the calling thread has been destroyed using cuCtxDestroy, or is a primary context which has not yet been initialized.
/// </summary>
[PublicAPI]
public class CudaContextIsDestroyedException : CudaException
{
    private const string DefaultMessage =
        "This error indicates that the context current to the calling thread has been destroyed using cuCtxDestroy, or is a primary context which has not yet been initialized.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaContextIsDestroyedException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the context current to the calling thread has been destroyed using cuCtxDestroy, or is a primary context which has not yet been initialized.
    /// </remarks>
    public CudaContextIsDestroyedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaContextIsDestroyedException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the context current to the calling thread has been destroyed using cuCtxDestroy, or is a primary context which has not yet been initialized.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaContextIsDestroyedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
