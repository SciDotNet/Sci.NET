// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.Interop;

/// <summary>
/// Enumerates the status code responses from the native libraries.
/// </summary>
[PublicAPI]
public enum SdnStatusCode
{
    /// <summary>
    /// Unknown status code.
    /// </summary>
    SdnStatusUnknown = 0,

    /// <summary>
    /// The operation was successful.
    /// </summary>
    SdnSuccess = 1,

    /// <summary>
    /// The native library was not initialized.
    /// </summary>
    SdnNotInitialized = 2,

    /// <summary>
    /// An internal operation failed.
    /// </summary>
    SdnInternalError = 3,
}