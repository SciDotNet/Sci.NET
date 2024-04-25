// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common;

namespace Sci.NET.Accelerators.Rewriter.SpecialInstructions;

/// <summary>
/// Represents a field of a <see cref="Dim3"/> value.
/// </summary>
[PublicAPI]
public enum Dim3Field
{
    /// <summary>
    /// No field.
    /// </summary>
    None = 0,

    /// <summary>
    /// The X field.
    /// </summary>
    X = 1,

    /// <summary>
    /// The Y field.
    /// </summary>
    Y = 3,

    /// <summary>
    /// The Z field.
    /// </summary>
    Z = 4
}