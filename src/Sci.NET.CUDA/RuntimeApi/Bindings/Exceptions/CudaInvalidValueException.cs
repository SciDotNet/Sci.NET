// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.
/// </summary>
[PublicAPI]
public class CudaInvalidValueException : CudaException
{
    private const string DefaultMessage =
        "This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidValueException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.
    /// </remarks>
    public CudaInvalidValueException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidValueException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidValueException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
