// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that a kernel launch is requesting resources that can never be satisfied by the current device. Requesting more shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks. See cudaDeviceProp for more device limitations.
/// </summary>
[PublicAPI]
public class CudaInvalidConfigurationException : CudaException
{
    private const string DefaultMessage =
        "This indicates that a kernel launch is requesting resources that can never be satisfied by the current device. Requesting more shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks. See cudaDeviceProp for more device limitations.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidConfigurationException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a kernel launch is requesting resources that can never be satisfied by the current device. Requesting more shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks. See cudaDeviceProp for more device limitations.
    /// </remarks>
    public CudaInvalidConfigurationException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidConfigurationException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a kernel launch is requesting resources that can never be satisfied by the current device. Requesting more shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks. See cudaDeviceProp for more device limitations.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidConfigurationException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
