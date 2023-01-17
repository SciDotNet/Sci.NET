// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// The device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
/// </summary>
[PublicAPI]
public class CudaMisalignedAddressException : CudaException
{
    private const string DefaultMessage =
        "The device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMisalignedAddressException"/> class.
    /// </summary>
    /// <remarks>
    /// The device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
    /// </remarks>
    public CudaMisalignedAddressException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMisalignedAddressException"/> class.
    /// </summary>
    /// <remarks>
    /// The device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaMisalignedAddressException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
