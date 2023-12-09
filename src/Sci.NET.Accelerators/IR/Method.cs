// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// Represents a method in the IR.
/// </summary>
[PublicAPI]
public class Method
{
    /// <summary>
    /// Gets the parameters of the method.
    /// </summary>
    public ICollection<Parameter> Parameters { get; init; }

    /// <summary>
    /// Gets the local variables of the method.
    /// </summary>
    public ICollection<LocalVariable> LocalVariables { get; init; }
}