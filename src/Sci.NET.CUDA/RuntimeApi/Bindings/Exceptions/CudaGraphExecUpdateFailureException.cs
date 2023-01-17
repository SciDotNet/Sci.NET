// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.
/// </summary>
[PublicAPI]
public class CudaGraphExecUpdateFailureException : CudaException
{
    private const string DefaultMessage =
        "This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaGraphExecUpdateFailureException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.
    /// </remarks>
    public CudaGraphExecUpdateFailureException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaGraphExecUpdateFailureException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaGraphExecUpdateFailureException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
