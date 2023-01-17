// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// An assert triggered in device code during kernel execution. The device cannot be used again. All existing allocations are invalid. To continue using CUDA, the process must be terminated and relaunched.
/// </summary>
[PublicAPI]
public class CudaAssertException : CudaException
{
    private const string DefaultMessage =
        "An assert triggered in device code during kernel execution. The device cannot be used again. All existing allocations are invalid. To continue using CUDA, the process must be terminated and relaunched.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaAssertException"/> class.
    /// </summary>
    /// <remarks>
    /// An assert triggered in device code during kernel execution. The device cannot be used again. All existing allocations are invalid. To continue using CUDA, the process must be terminated and relaunched.
    /// </remarks>
    public CudaAssertException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaAssertException"/> class.
    /// </summary>
    /// <remarks>
    /// An assert triggered in device code during kernel execution. The device cannot be used again. All existing allocations are invalid. To continue using CUDA, the process must be terminated and relaunched.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaAssertException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
