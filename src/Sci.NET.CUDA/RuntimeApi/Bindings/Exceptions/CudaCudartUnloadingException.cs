// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that a CUDA Runtime API call cannot be executed because it is being called during process shut down, at a point in time after CUDA driver has been unloaded.
/// </summary>
[PublicAPI]
public class CudaCudartUnloadingException : CudaException
{
    private const string DefaultMessage =
        "This indicates that a CUDA Runtime API call cannot be executed because it is being called during process shut down, at a point in time after CUDA driver has been unloaded.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaCudartUnloadingException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a CUDA Runtime API call cannot be executed because it is being called during process shut down, at a point in time after CUDA driver has been unloaded.
    /// </remarks>
    public CudaCudartUnloadingException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaCudartUnloadingException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a CUDA Runtime API call cannot be executed because it is being called during process shut down, at a point in time after CUDA driver has been unloaded.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaCudartUnloadingException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
