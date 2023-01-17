// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// The device encountered an illegal instruction during kernel execution This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
/// </summary>
[PublicAPI]
public class CudaIllegalInstructionException : CudaException
{
    private const string DefaultMessage =
        "The device encountered an illegal instruction during kernel execution This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaIllegalInstructionException"/> class.
    /// </summary>
    /// <remarks>
    /// The device encountered an illegal instruction during kernel execution This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
    /// </remarks>
    public CudaIllegalInstructionException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaIllegalInstructionException"/> class.
    /// </summary>
    /// <remarks>
    /// The device encountered an illegal instruction during kernel execution This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaIllegalInstructionException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
