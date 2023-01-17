// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the file specified was not found.
/// </summary>
[PublicAPI]
public class CudaFileNotFoundException : CudaException
{
    private const string DefaultMessage = "This indicates that the file specified was not found.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaFileNotFoundException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the file specified was not found.
    /// </remarks>
    public CudaFileNotFoundException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaFileNotFoundException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the file specified was not found.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaFileNotFoundException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
