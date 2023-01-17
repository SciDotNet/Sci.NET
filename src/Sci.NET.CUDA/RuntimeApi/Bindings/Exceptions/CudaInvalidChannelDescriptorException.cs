// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that the channel descriptor passed to the API call is not valid. This occurs if the format is not one of the formats specified by cudaChannelFormatKind, or if one of the dimensions is invalid.
/// </summary>
[PublicAPI]
public class CudaInvalidChannelDescriptorException : CudaException
{
    private const string DefaultMessage =
        "This indicates that the channel descriptor passed to the API call is not valid. This occurs if the format is not one of the formats specified by cudaChannelFormatKind, or if one of the dimensions is invalid.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidChannelDescriptorException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the channel descriptor passed to the API call is not valid. This occurs if the format is not one of the formats specified by cudaChannelFormatKind, or if one of the dimensions is invalid.
    /// </remarks>
    public CudaInvalidChannelDescriptorException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidChannelDescriptorException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that the channel descriptor passed to the API call is not valid. This occurs if the format is not one of the formats specified by cudaChannelFormatKind, or if one of the dimensions is invalid.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidChannelDescriptorException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
