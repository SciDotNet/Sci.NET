// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// Represents parameter for a method.
/// </summary>
[PublicAPI]
public class Parameter
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Parameter"/> class.
    /// </summary>
    /// <param name="type">The type of the parameter.</param>
    /// <param name="index">The index of the parameter.</param>
    public Parameter(Type type, int index)
    {
        Type = type;
        Index = index;
    }

    /// <summary>
    /// Gets the type of the parameter.
    /// </summary>
    public Type Type { get; init; }

    /// <summary>
    /// Gets the index of the parameter.
    /// </summary>
    public int Index { get; init; }
}