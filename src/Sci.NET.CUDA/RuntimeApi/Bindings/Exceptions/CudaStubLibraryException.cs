// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the CUDA driver that the application has loaded is a stub library. Applications that run with the stub rather than a real driver loaded will result in CUDA API returning this error.
/// </summary>
[PublicAPI]
public class CudaStubLibraryException : CudaException
{
    private const string DefaultMessage =
        "This indicates that the CUDA driver that the application has loaded is a stub library. Applications that run with the stub rather than a real driver loaded will result in CUDA API returning this error.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStubLibraryException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the CUDA driver that the application has loaded is a stub library. Applications that run with the stub rather than a real driver loaded will result in CUDA API returning this error.
    /// </remarks>
    public CudaStubLibraryException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaStubLibraryException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the CUDA driver that the application has loaded is a stub library. Applications that run with the stub rather than a real driver loaded will result in CUDA API returning this error.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaStubLibraryException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
