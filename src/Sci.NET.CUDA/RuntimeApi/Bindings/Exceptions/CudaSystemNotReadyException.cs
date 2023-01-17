// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide.
/// </summary>
[PublicAPI]
public class CudaSystemNotReadyException : CudaException
{
    private const string DefaultMessage =
        "This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaSystemNotReadyException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide.
    /// </remarks>
    public CudaSystemNotReadyException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaSystemNotReadyException"/> class.
    /// </summary>
    /// <remarks>
    /// This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaSystemNotReadyException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
