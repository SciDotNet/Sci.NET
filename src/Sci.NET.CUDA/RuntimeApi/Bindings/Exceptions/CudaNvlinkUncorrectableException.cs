// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that an uncorrectable NVLink error was detected during the execution.
/// </summary>
[PublicAPI]
public class CudaNvlinkUncorrectableException : CudaException
{
    private const string DefaultMessage =
        "This indicates that an uncorrectable NVLink error was detected during the execution.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaNvlinkUncorrectableException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that an uncorrectable NVLink error was detected during the execution.
    /// </remarks>
    public CudaNvlinkUncorrectableException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaNvlinkUncorrectableException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that an uncorrectable NVLink error was detected during the execution.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaNvlinkUncorrectableException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
