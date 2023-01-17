// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates a kernel launch that uses an incompatible texturing mode.
/// </summary>
[PublicAPI]
public class CudaLaunchIncompatibleTexturingException : CudaException
{
    private const string DefaultMessage =
        "This error indicates a kernel launch that uses an incompatible texturing mode.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaLaunchIncompatibleTexturingException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates a kernel launch that uses an incompatible texturing mode.
    /// </remarks>
    public CudaLaunchIncompatibleTexturingException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaLaunchIncompatibleTexturingException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates a kernel launch that uses an incompatible texturing mode.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaLaunchIncompatibleTexturingException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
