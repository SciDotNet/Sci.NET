// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that a launch did not occur because it did not have appropriate resources. Although this error is similar to cudaErrorInvalidConfiguration, this error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count.
/// </summary>
[PublicAPI]
public class CudaLaunchOutOfResourcesException : CudaException
{
    private const string DefaultMessage =
        "This indicates that a launch did not occur because it did not have appropriate resources. Although this error is similar to cudaErrorInvalidConfiguration, this error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaLaunchOutOfResourcesException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a launch did not occur because it did not have appropriate resources. Although this error is similar to cudaErrorInvalidConfiguration, this error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count.
    /// </remarks>
    public CudaLaunchOutOfResourcesException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaLaunchOutOfResourcesException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a launch did not occur because it did not have appropriate resources. Although this error is similar to cudaErrorInvalidConfiguration, this error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaLaunchOutOfResourcesException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
