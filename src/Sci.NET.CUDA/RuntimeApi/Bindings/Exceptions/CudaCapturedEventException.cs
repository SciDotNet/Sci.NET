// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// The operation is not permitted on an event which was last recorded in a capturing stream.
/// </summary>
[PublicAPI]
public class CudaCapturedEventException : CudaException
{
    private const string DefaultMessage =
        "The operation is not permitted on an event which was last recorded in a capturing stream.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaCapturedEventException"/> class.
    /// </summary>
    /// <remarks>
    /// The operation is not permitted on an event which was last recorded in a capturing stream.
    /// </remarks>
    public CudaCapturedEventException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaCapturedEventException"/> class.
    /// </summary>
    /// <remarks>
    /// The operation is not permitted on an event which was last recorded in a capturing stream.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaCapturedEventException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
