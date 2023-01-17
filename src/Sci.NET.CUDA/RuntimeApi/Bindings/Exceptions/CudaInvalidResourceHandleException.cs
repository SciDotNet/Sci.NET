// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like cudaStream_t and cudaEvent_t.
/// </summary>
[PublicAPI]
public class CudaInvalidResourceHandleException : CudaException
{
    private const string DefaultMessage =
        "This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like cudaStream_t and cudaEvent_t.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidResourceHandleException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like cudaStream_t and cudaEvent_t.
    /// </remarks>
    public CudaInvalidResourceHandleException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaInvalidResourceHandleException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like cudaStream_t and cudaEvent_t.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaInvalidResourceHandleException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
