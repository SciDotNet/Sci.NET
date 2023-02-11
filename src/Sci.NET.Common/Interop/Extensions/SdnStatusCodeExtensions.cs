// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace Sci.NET.Common.Interop.Extensions;

/// <summary>
/// Extension methods for <see cref="SdnStatusCode"/>.
/// </summary>
[PublicAPI]
public static class SdnStatusCodeExtensions
{
    /// <summary>
    /// Throws an exception if the status code does not indicate success.
    /// </summary>
    /// <param name="status">The status code returned by the native method.</param>
    /// <param name="callerMemberName">The caller member name.</param>
    /// <exception cref="InvalidOperationException">The native method did not indicate success.</exception>
    [SuppressMessage("Roslynator", "RCS1069:Remove unnecessary case label.", Justification = "Readability.")]
    public static void ThrowOnError(this SdnStatusCode status, [CallerMemberName] string? callerMemberName = null)
    {
        switch (status)
        {
            case SdnStatusCode.SdnSuccess:
                break;
            case SdnStatusCode.SdnNotInitialized:
                throw new InvalidOperationException(
                    $"A call was made to '{callerMemberName}' before the library was initialized.");
            case SdnStatusCode.SdnInternalError:
                throw new InvalidOperationException($"An internal error occurred in the call to '{callerMemberName}'.");
            case SdnStatusCode.SdnStatusUnknown:
            default:
                throw new InvalidOperationException(
                    $"The status code of the call to '{callerMemberName}' does not indicate success but is unknown.");
        }
    }
}