// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the JIT compilation was disabled. The JIT compilation compiles PTX. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.
/// </summary>
[PublicAPI]
public class CudaJitCompilationDisabledException : CudaException
{
    private const string DefaultMessage =
        "This indicates that the JIT compilation was disabled. The JIT compilation compiles PTX. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaJitCompilationDisabledException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the JIT compilation was disabled. The JIT compilation compiles PTX. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.
    /// </remarks>
    public CudaJitCompilationDisabledException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaJitCompilationDisabledException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the JIT compilation was disabled. The JIT compilation compiles PTX. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaJitCompilationDisabledException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
