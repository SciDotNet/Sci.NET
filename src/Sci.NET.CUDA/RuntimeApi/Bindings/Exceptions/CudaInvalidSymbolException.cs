// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the symbol name/identifier passed to the API call is not a valid name or identifier.
/// </summary>
[PublicAPI]
public class CudaInvalidSymbolException : CudaException
{
    private const string DefaultMessage =
        "This indicates that the symbol name/identifier passed to the API call is not a valid name or identifier.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidSymbolException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the symbol name/identifier passed to the API call is not a valid name or identifier.
    /// </remarks>
    public CudaInvalidSymbolException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidSymbolException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the symbol name/identifier passed to the API call is not a valid name or identifier.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidSymbolException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
