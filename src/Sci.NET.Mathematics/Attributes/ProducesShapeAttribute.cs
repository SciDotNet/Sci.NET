// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Attributes;

/// <summary>
/// An attribute describing the resultant shape of an <see cref="ITensor{TNumber}"/>
/// for a parameter of a function.
/// </summary>
[PublicAPI]
[AttributeUsage(AttributeTargets.Parameter | AttributeTargets.Method)]
[ExcludeFromCodeCoverage]
public sealed class ProducesShapeAttribute : Attribute
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ProducesShapeAttribute"/> class.
    /// </summary>
    /// <param name="shape">The shape of the <see cref="ITensor{TNumber}"/>.</param>
    public ProducesShapeAttribute(string shape)
    {
        Shape = shape;
    }

    /// <summary>
    /// Gets the expression describing the expected shape.
    /// </summary>
    public string Shape { get; }
}