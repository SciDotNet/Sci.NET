// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.CuBLAS.Exceptions;

/// <summary>
/// An exception thrown when accessing a license error occurs.
/// </summary>
[PublicAPI]
public class CublasLicenseErrorException : Exception
{
    private const string DefaultMessage =
        "The functionality requested requires some license and an error was detected when trying to check the current licensing.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CublasLicenseErrorException"/> class.
    /// </summary>
    public CublasLicenseErrorException()
        : base(DefaultMessage)
    {
    }
}