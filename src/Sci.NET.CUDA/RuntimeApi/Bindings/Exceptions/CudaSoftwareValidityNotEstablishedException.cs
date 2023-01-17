// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// By default, the CUDA runtime may perform a minimal set of self-tests, as well as CUDA driver tests, to establish the validity of both. Introduced in CUDA 11.2, this error return indicates that at least one of these tests has failed and the validity of either the runtime or the driver could not be established.
/// </summary>
[PublicAPI]
public class CudaSoftwareValidityNotEstablishedException : CudaException
{
    private const string DefaultMessage =
        "By default, the CUDA runtime may perform a minimal set of self-tests, as well as CUDA driver tests, to establish the validity of both. Introduced in CUDA 11.2, this error return indicates that at least one of these tests has failed and the validity of either the runtime or the driver could not be established.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaSoftwareValidityNotEstablishedException"/> class.
    /// </summary>
    /// <remarks>
    /// By default, the CUDA runtime may perform a minimal set of self-tests, as well as CUDA driver tests, to establish the validity of both. Introduced in CUDA 11.2, this error return indicates that at least one of these tests has failed and the validity of either the runtime or the driver could not be established.
    /// </remarks>
    public CudaSoftwareValidityNotEstablishedException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaSoftwareValidityNotEstablishedException"/> class.
    /// </summary>
    /// <remarks>
    /// By default, the CUDA runtime may perform a minimal set of self-tests, as well as CUDA driver tests, to establish the validity of both. Introduced in CUDA 11.2, this error return indicates that at least one of these tests has failed and the validity of either the runtime or the driver could not be established.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaSoftwareValidityNotEstablishedException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
