// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// The device function being invoked (usually via cudaLaunchKernel()) was not previously configured via the cudaConfigureCall() function.
/// </summary>
[PublicAPI]
public class CudaMissingConfigurationException : CudaException
{
    private const string DefaultMessage =
        "The device function being invoked (usually via cudaLaunchKernel()) was not previously configured via the cudaConfigureCall() function.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMissingConfigurationException"/> class.
    /// </summary>
    /// <remarks>
    /// The device function being invoked (usually via cudaLaunchKernel()) was not previously configured via the cudaConfigureCall() function.
    /// </remarks>
    public CudaMissingConfigurationException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaMissingConfigurationException"/> class.
    /// </summary>
    /// <remarks>
    /// The device function being invoked (usually via cudaLaunchKernel()) was not previously configured via the cudaConfigureCall() function.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaMissingConfigurationException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
