// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Text;
using Sci.NET.Accelerators.IR;

namespace Sci.NET.Accelerators.UnitTests.Extensions;

public static class BasicBlockExtensions
{
    public static string ConvertToString(this IEnumerable<BasicBlock> blocks)
    {
        var sb = new StringBuilder();
        var instructionCount = 0;

        foreach (var block in blocks)
        {
#pragma warning disable CA1305
            _ = sb
                .Append("Block ")
                .AppendFormat("{0:X4}", block.StartOffset)
                .Append('-')
                .AppendFormat("{0:X4}", block.EndOffset)
                .AppendLine();
#pragma warning restore CA1305

            foreach (var instruction in block.Instructions)
            {
                sb.AppendLine(instruction.ToString());
                instructionCount++;
            }

            sb.AppendLine();
        }

#pragma warning disable CA1305
        sb.Append("Total instructions: ").Append(instructionCount).AppendLine();
#pragma warning restore CA1305

        return sb.ToString();
    }
}