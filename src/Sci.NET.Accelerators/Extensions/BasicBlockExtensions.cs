// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Text;
using Sci.NET.Accelerators.IR;

namespace Sci.NET.Accelerators.Extensions;

/// <summary>
/// Extension methods for <see cref="BasicBlock"/>.
/// </summary>
[PublicAPI]
public static class BasicBlockExtensions
{
    /// <summary>
    /// Converts a collection <see cref="BasicBlock"/> to an IR string.
    /// </summary>
    /// <param name="basicBlocks">The collection of <see cref="BasicBlock"/> to convert.</param>
    /// <returns>The IR string representation of the <see cref="BasicBlock"/>s.</returns>
    public static string ToIR(this IEnumerable<BasicBlock> basicBlocks)
    {
        var sb = new StringBuilder();

        foreach (var block in basicBlocks)
        {
            _ = block.WriteToIrString(sb, 0);
        }

        return sb.ToString();
    }
}