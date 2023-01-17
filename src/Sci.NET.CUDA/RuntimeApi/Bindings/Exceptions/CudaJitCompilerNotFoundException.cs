// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the PTX JIT compiler library was not found. The JIT Compiler library is used for PTX compilation. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.
/// </summary>
[PublicAPI]
public class CudaJitCompilerNotFoundException : CudaException
{
    private const string DefaultMessage =
        "This indicates that the PTX JIT compiler library was not found. The JIT Compiler library is used for PTX compilation. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaJitCompilerNotFoundException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the PTX JIT compiler library was not found. The JIT Compiler library is used for PTX compilation. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.
    /// </remarks>
    public CudaJitCompilerNotFoundException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaJitCompilerNotFoundException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the PTX JIT compiler library was not found. The JIT Compiler library is used for PTX compilation. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaJitCompilerNotFoundException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
