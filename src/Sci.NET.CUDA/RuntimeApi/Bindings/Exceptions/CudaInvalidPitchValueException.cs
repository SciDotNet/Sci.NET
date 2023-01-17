// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that one or more of the pitch-related parameters passed to the API call is not within the acceptable range for pitch.
/// </summary>
[PublicAPI]
public class CudaInvalidPitchValueException : CudaException
{
    private const string DefaultMessage =
        "This indicates that one or more of the pitch-related parameters passed to the API call is not within the acceptable range for pitch.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidPitchValueException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that one or more of the pitch-related parameters passed to the API call is not within the acceptable range for pitch.
    /// </remarks>
    public CudaInvalidPitchValueException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidPitchValueException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that one or more of the pitch-related parameters passed to the API call is not within the acceptable range for pitch.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidPitchValueException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
