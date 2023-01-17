// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that multiple global or constant variables (across separate CUDA source files in the application) share the same string name.
/// </summary>
[PublicAPI]
public class CudaDuplicateVariableNameException : CudaException
{
    private const string DefaultMessage =
        "This indicates that multiple global or constant variables (across separate CUDA source files in the application) share the same string name.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaDuplicateVariableNameException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that multiple global or constant variables (across separate CUDA source files in the application) share the same string name.
    /// </remarks>
    public CudaDuplicateVariableNameException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaDuplicateVariableNameException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that multiple global or constant variables (across separate CUDA source files in the application) share the same string name.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaDuplicateVariableNameException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
