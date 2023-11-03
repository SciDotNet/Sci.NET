// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.Native.Extensions;

/// <summary>
/// Extensions for <see cref="SdnApiStatusCode" />.
/// </summary>
[PublicAPI]
public static class SdnApiStatusCodeExtensions
{
    /// <summary>
    /// Throws an exception if the status code is not <see cref="SdnApiStatusCode.Success" />.
    /// </summary>
    /// <param name="code">The status code.</param>
    /// <exception cref="InvalidOperationException">Thrown if the status code is not <see cref="SdnApiStatusCode.Success" />.</exception>
    public static void Guard(this SdnApiStatusCode code)
    {
        if (code != SdnApiStatusCode.Success)
        {
            throw new InvalidOperationException($"SDN API returned status code {code}.");
        }
    }
}