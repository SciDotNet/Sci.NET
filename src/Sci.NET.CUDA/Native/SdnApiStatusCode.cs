// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.Native;

/// <summary>
/// SDN API status codes.
/// </summary>
public enum SdnApiStatusCode
{
    /// <summary>
    /// Unknown status code.
    /// </summary>
    StatusUnknown = 0,

    /// <summary>
    /// Success.
    /// </summary>
    Success = 1,

    /// <summary>
    /// Not initialized.
    /// </summary>
    NotInitialized = 2,

    /// <summary>
    /// Invalid value.
    /// </summary>
    InvalidValue = 3,

    /// <summary>
    /// Internal error.
    /// </summary>
    InternalError = 4,
}