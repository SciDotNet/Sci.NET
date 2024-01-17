// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Text;

namespace Sci.NET.Accelerators.IR.Rewriter;

/// <summary>
/// Represents a method.
/// </summary>
[PublicAPI]
public class SsaMethodBody
{
    /// <summary>
    /// Initializes a new instance of the <see cref="SsaMethodBody"/> class.
    /// </summary>
    /// <param name="instructions">The instructions of the method.</param>
    public SsaMethodBody(ICollection<SsaInstruction> instructions)
    {
        Instructions = instructions;
    }

    /// <summary>
    /// Gets the instructions of the method.
    /// </summary>
    public ICollection<SsaInstruction> Instructions { get; }

    /// <inheritdoc />
    public override string ToString()
    {
        var sb = new StringBuilder();

        foreach (var instruction in Instructions)
        {
            _ = sb.Append(instruction.ToString());
        }

        return sb.ToString();
    }
}