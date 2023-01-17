// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than cudaSuccess (which indicates completion). Calls that may return this value include cudaEventQuery() and cudaStreamQuery().
/// </summary>
[PublicAPI]
public class CudaNotReadyException : CudaException
{
    private const string DefaultMessage =
        "This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than cudaSuccess (which indicates completion). Calls that may return this value include cudaEventQuery() and cudaStreamQuery().";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaNotReadyException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than cudaSuccess (which indicates completion). Calls that may return this value include cudaEventQuery() and cudaStreamQuery().
    /// </remarks>
    public CudaNotReadyException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaNotReadyException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than cudaSuccess (which indicates completion). Calls that may return this value include cudaEventQuery() and cudaStreamQuery().
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaNotReadyException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
