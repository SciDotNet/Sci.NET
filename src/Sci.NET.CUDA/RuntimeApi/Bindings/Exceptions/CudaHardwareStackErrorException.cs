// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// Device encountered an error in the call stack during kernel execution, possibly due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
/// </summary>
[PublicAPI]
public class CudaHardwareStackErrorException : CudaException
{
    private const string DefaultMessage =
        "Device encountered an error in the call stack during kernel execution, possibly due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaHardwareStackErrorException"/> class.
    /// </summary>
    /// <remarks>
    /// Device encountered an error in the call stack during kernel execution, possibly due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
    /// </remarks>
    public CudaHardwareStackErrorException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaHardwareStackErrorException"/> class.
    /// </summary>
    /// <remarks>
    /// Device encountered an error in the call stack during kernel execution, possibly due to stack corruption or exceeding the stack count limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaHardwareStackErrorException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
