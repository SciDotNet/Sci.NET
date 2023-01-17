// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.CuBLAS.Exceptions;

/// <summary>
/// An exception that is thrown when a cuBLAS operation was passed an invalid value.
/// </summary>
[PublicAPI]
public class CublasInvalidValueException : Exception
{
    private const string DefaultMessage = "An invalid value was passed to a cuBLAS operation.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CublasInvalidValueException"/> class.
    /// </summary>
    public CublasInvalidValueException()
        : base(DefaultMessage)
    {
    }
}