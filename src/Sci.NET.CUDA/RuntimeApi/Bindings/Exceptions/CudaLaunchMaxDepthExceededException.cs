// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates that a device runtime grid launch did not occur because the depth of the child grid would exceed the maximum supported number of nested grid launches.
/// </summary>
[PublicAPI]
public class CudaLaunchMaxDepthExceededException : CudaException
{
    private const string DefaultMessage =
        "This error indicates that a device runtime grid launch did not occur because the depth of the child grid would exceed the maximum supported number of nested grid launches.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaLaunchMaxDepthExceededException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that a device runtime grid launch did not occur because the depth of the child grid would exceed the maximum supported number of nested grid launches.
    /// </remarks>
    public CudaLaunchMaxDepthExceededException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaLaunchMaxDepthExceededException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that a device runtime grid launch did not occur because the depth of the child grid would exceed the maximum supported number of nested grid launches.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaLaunchMaxDepthExceededException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
