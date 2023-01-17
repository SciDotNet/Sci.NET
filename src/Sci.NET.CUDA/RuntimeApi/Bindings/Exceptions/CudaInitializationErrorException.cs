// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// The API call failed because the CUDA driver and runtime could not be initialized.
/// </summary>
[PublicAPI]
public class CudaInitializationErrorException : CudaException
{
    private const string DefaultMessage =
        "The API call failed because the CUDA driver and runtime could not be initialized.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInitializationErrorException"/> class.
    /// </summary>
    /// <remarks>
    /// The API call failed because the CUDA driver and runtime could not be initialized.
    /// </remarks>
    public CudaInitializationErrorException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInitializationErrorException"/> class.
    /// </summary>
    /// <remarks>
    /// The API call failed because the CUDA driver and runtime could not be initialized.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInitializationErrorException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
