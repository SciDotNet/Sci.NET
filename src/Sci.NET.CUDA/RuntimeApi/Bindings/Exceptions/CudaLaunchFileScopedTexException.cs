// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates that a grid launch did not occur because the kernel uses file-scoped textures which are unsupported by the device runtime. Kernels launched via the device runtime only support textures created with the Texture Object API's.
/// </summary>
[PublicAPI]
public class CudaLaunchFileScopedTexException : CudaException
{
    private const string DefaultMessage =
        "This error indicates that a grid launch did not occur because the kernel uses file-scoped textures which are unsupported by the device runtime. Kernels launched via the device runtime only support textures created with the Texture Object API's.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaLaunchFileScopedTexException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that a grid launch did not occur because the kernel uses file-scoped textures which are unsupported by the device runtime. Kernels launched via the device runtime only support textures created with the Texture Object API's.
    /// </remarks>
    public CudaLaunchFileScopedTexException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaLaunchFileScopedTexException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that a grid launch did not occur because the kernel uses file-scoped textures which are unsupported by the device runtime. Kernels launched via the device runtime only support textures created with the Texture Object API's.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaLaunchFileScopedTexException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
