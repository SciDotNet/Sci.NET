// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Attributes;

/// <summary>
/// An attribute describing the expected shapes of an <see cref="ITensor{TNumber}"/>
/// for a parameter of a function.
/// </summary>
[PublicAPI]
[AttributeUsage(AttributeTargets.Parameter)]
[ExcludeFromCodeCoverage]
public sealed class AssumesShapeAttribute : Attribute
{
    /// <summary>
    /// Initializes a new instance of the <see cref="AssumesShapeAttribute"/> class.
    /// </summary>
    /// <param name="shapeExpression">The expression describing the shape.</param>
    /// <param name="parameterName">The name of the parameter to describe.</param>
    public AssumesShapeAttribute(string shapeExpression, [CallerMemberName] string parameterName = "")
    {
        ParameterName = parameterName;
        ShapeExpression = shapeExpression;
    }

    /// <summary>
    /// Gets the name of the parameter to describe the shape of.
    /// </summary>
    public string ParameterName { get; }

    /// <summary>
    /// Gets the expression describing the expected shape.
    /// </summary>
    public string ShapeExpression { get; }
}