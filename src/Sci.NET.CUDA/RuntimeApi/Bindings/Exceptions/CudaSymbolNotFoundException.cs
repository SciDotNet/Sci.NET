// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, driver function names, texture names, and surface names.
/// </summary>
[PublicAPI]
public class CudaSymbolNotFoundException : CudaException
{
    private const string DefaultMessage =
        "This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, driver function names, texture names, and surface names.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaSymbolNotFoundException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, driver function names, texture names, and surface names.
    /// </remarks>
    public CudaSymbolNotFoundException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaSymbolNotFoundException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, driver function names, texture names, and surface names.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaSymbolNotFoundException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
