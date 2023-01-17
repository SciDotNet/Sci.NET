// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// A CUDA CudaException.
/// </summary>
[PublicAPI]
public abstract class CudaException : Exception
{
    /// <inheritdoc />
    protected CudaException(string message)
        : base(message)
    {
    }

    /// <inheritdoc />
    protected CudaException(string message, Exception innerCudaException)
        : base(message, innerCudaException)
    {
    }
}