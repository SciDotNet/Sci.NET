// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// The operation would have resulted in a merge of two independent capture sequences.
/// </summary>
[PublicAPI]
public class CudaStreamCaptureMergeException : CudaException
{
    private const string DefaultMessage =
        "The operation would have resulted in a merge of two independent capture sequences.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStreamCaptureMergeException"/> class.
    /// </summary>
    /// <remarks>
    /// The operation would have resulted in a merge of two independent capture sequences.
    /// </remarks>
    public CudaStreamCaptureMergeException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStreamCaptureMergeException"/> class.
    /// </summary>
    /// <remarks>
    /// The operation would have resulted in a merge of two independent capture sequences.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaStreamCaptureMergeException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
