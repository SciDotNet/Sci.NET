// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// Represents a local variable.
/// </summary>
[PublicAPI]
public class LocalVariable
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LocalVariable"/> class.
    /// </summary>
    /// <param name="type">The type of the local variable.</param>
    public LocalVariable(Type type)
    {
        Type = type;
    }

    /// <summary>
    /// Gets the type of the local variable.
    /// </summary>
    public Type Type { get; init; }
}