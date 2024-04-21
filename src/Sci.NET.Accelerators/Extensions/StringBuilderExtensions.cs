// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Text;
using Sci.NET.Accelerators.IR;

namespace Sci.NET.Accelerators.Extensions;

internal static class StringBuilderExtensions
{
    public static StringBuilder AppendIndent(this StringBuilder builder, int indentLevel, int indentSize = 2)
    {
        for (var i = 0; i < indentLevel; i++)
        {
            _ = builder.Append(' ', indentSize);
        }

        return builder;
    }

    public static StringBuilder AppendWritable(this StringBuilder builder, IIrWritable writable)
    {
        return writable.WriteToIrString(builder);
    }
}