// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that a link to a shared object failed to resolve.
/// </summary>
[PublicAPI]
public class CudaSharedObjectSymbolNotFoundException : CudaException
{
    private const string DefaultMessage = "This indicates that a link to a shared object failed to resolve.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaSharedObjectSymbolNotFoundException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a link to a shared object failed to resolve.
    /// </remarks>
    public CudaSharedObjectSymbolNotFoundException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaSharedObjectSymbolNotFoundException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a link to a shared object failed to resolve.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaSharedObjectSymbolNotFoundException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
