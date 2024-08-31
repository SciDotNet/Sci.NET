// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace Sci.NET.Common.Attributes;

/// <summary>
/// An attribute that represents a mathematical expression for documentation purposes.
/// </summary>
[AttributeUsage(AttributeTargets.Method, Inherited = true, AllowMultiple = true)]
[ExcludeFromCodeCoverage]
public sealed class MathematicExpressionAttribute : Attribute
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MathematicExpressionAttribute"/> class.
    /// </summary>
    /// <param name="abstractionIndex">The abstraction index, 0 being the most abstract form.</param>
    /// <param name="latex">The LaTeX expression.</param>
    public MathematicExpressionAttribute(int abstractionIndex, string latex)
    {
        AbstractionIndex = abstractionIndex;
        Latex = latex;
    }

    /// <summary>
    /// Gets the LaTeX expression.
    /// </summary>
    public string Latex { get; }

    /// <summary>
    /// Gets the abstraction index.
    /// </summary>
    public int AbstractionIndex { get; }
}