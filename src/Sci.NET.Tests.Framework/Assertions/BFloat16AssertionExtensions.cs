// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Numerics;

namespace Sci.NET.Tests.Framework.Assertions;

/// <summary>
/// Assertion extensions for <see cref="BFloat16" />.
/// </summary>
[PublicAPI]
public static class BFloat16AssertionExtensions
{
    /// <summary>
    /// Extension method to create a <see cref="BFloat16Assertions" /> object for the given <see cref="BFloat16" />.
    /// </summary>
    /// <param name="bfloat16">The bfloat16 to create assertions for.</param>
    /// <returns>A <see cref="BFloat16Assertions" /> object for the given <see cref="BFloat16" />.</returns>
    public static BFloat16Assertions Should(this BFloat16 bfloat16)
    {
        return new(bfloat16);
    }
}