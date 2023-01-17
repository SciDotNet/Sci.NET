// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.CuBLAS.Types;

/// <summary>
/// Enumerates the status codes returned by the cuBLAS library.
/// </summary>
[PublicAPI]
public enum CublasStatus
{
    /// <summary>
    /// Indicates the operation completed successfully.
    /// </summary>
    CublasSuccess = 0,

    /// <summary>
    /// Indicates that the cuBLAS library was not initialized.
    /// </summary>
    CublasNotInitialized = 1,

    /// <summary>
    /// Indicates that an allocation failed.
    /// </summary>
    CublasAllocFailed = 3,

    /// <summary>
    /// Indicates that an invalid value was passed to the function.
    /// </summary>
    CublasInvalidValue = 7,

    /// <summary>
    /// Indicates that a required feature is not available on the device.
    /// </summary>
    CublasArchMismatch = 8,

    /// <summary>
    /// Indicates that a GPU memory access failed.
    /// </summary>
    CublasMappingError = 11,

    /// <summary>
    /// Indicates that an execution failed.
    /// </summary>
    CublasExecutionFailed = 13,

    /// <summary>
    /// Indicates that an internal operation failed.
    /// </summary>
    CublasInternalError = 14,

    /// <summary>
    /// Indicates that the requested functionality is not available.
    /// </summary>
    CublasNotSupported = 15,

    /// <summary>
    /// Indicates that the functionality requested requires a license
    /// and an error was detected when trying to check the current licensing.
    /// </summary>
    CublasLicenseError = 16
}