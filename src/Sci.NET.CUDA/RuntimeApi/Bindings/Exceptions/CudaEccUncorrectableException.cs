// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that an uncorrectable ECC error was detected during execution.
/// </summary>
[PublicAPI]
public class CudaEccUncorrectableException : CudaException
{
    private const string DefaultMessage =
        "This indicates that an uncorrectable ECC error was detected during execution.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaEccUncorrectableException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that an uncorrectable ECC error was detected during execution.
    /// </remarks>
    public CudaEccUncorrectableException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaEccUncorrectableException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that an uncorrectable ECC error was detected during execution.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaEccUncorrectableException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
