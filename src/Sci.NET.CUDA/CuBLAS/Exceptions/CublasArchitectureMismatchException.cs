// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.CuBLAS.Exceptions;

/// <summary>
/// An exception thrown when the cuBLAS API reports that a feature
/// is absent from the device architecture.
/// </summary>
[PublicAPI]
public class CublasArchitectureMismatchException : Exception
{
    private const string DefaultMessage =
        "A required feature for the cuBLAS operation is absent from the device architecture.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CublasArchitectureMismatchException"/> class.
    /// </summary>
    public CublasArchitectureMismatchException()
        : base(DefaultMessage)
    {
    }
}