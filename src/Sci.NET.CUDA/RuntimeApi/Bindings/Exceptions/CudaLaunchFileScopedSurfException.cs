// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates that a grid launch did not occur because the kernel uses file-scoped surfaces which are unsupported by the device runtime. Kernels launched via the device runtime only support surfaces created with the Surface Object API's.
/// </summary>
[PublicAPI]
public class CudaLaunchFileScopedSurfException : CudaException
{
    private const string DefaultMessage =
        "This error indicates that a grid launch did not occur because the kernel uses file-scoped surfaces which are unsupported by the device runtime. Kernels launched via the device runtime only support surfaces created with the Surface Object API's.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaLaunchFileScopedSurfException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that a grid launch did not occur because the kernel uses file-scoped surfaces which are unsupported by the device runtime. Kernels launched via the device runtime only support surfaces created with the Surface Object API's.
    /// </remarks>
    public CudaLaunchFileScopedSurfException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaLaunchFileScopedSurfException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that a grid launch did not occur because the kernel uses file-scoped surfaces which are unsupported by the device runtime. Kernels launched via the device runtime only support surfaces created with the Surface Object API's.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaLaunchFileScopedSurfException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
