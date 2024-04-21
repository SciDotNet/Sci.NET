// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Text;

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// An object that can be written to IR string representation.
/// </summary>
[PublicAPI]
public interface IIrWritable
{
    /// <summary>
    /// Writes the IR string representation to the string builder.
    /// </summary>
    /// <param name="builder">The string builder to write to.</param>
    /// <returns>The instance of the string builder.</returns>
    public StringBuilder WriteToIrString(StringBuilder builder);
}