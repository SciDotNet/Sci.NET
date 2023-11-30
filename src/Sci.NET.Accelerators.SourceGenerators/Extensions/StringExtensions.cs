// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.SourceGenerators.Extensions;

internal static class StringExtensions
{
    public static string EnsureEndsWith(this string str, string ending)
    {
        return str.EndsWith(ending, StringComparison.InvariantCulture) ? str : $"{str}{ending}";
    }
}