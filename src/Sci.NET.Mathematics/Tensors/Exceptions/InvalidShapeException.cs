// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace Sci.NET.Mathematics.Tensors.Exceptions;

/// <summary>
/// The exception that is thrown when a <see cref="ITensor{TNumber}"/> has an invalid shape.
/// </summary>
[PublicAPI]
[ExcludeFromCodeCoverage]
public class InvalidShapeException : Exception
{
    /// <summary>
    /// Initializes a new instance of the <see cref="InvalidShapeException"/> class.
    /// </summary>
    /// <param name="handler">The interpolated string handler.</param>
    public InvalidShapeException([InterpolatedStringHandlerArgument] InvalidShapeInterpolatedStringHandler handler)
        : base($"The given shape is invalid. {handler.GetFormattedString()}.")
    {
        GivenShapes = handler.GetShapes();
    }

    /// <summary>
    /// Gets the given shape.
    /// </summary>
    public IEnumerable<Shape> GivenShapes { get; }

    /// <summary>
    /// Throws an exception if the shapes are different.
    /// </summary>
    /// <param name="shapes">The shapes to compare.</param>
    /// <exception cref="InvalidShapeException">Throws when the shapes are different.</exception>
    [StackTraceHidden]
    public static void ThrowIfDifferentShape(params Shape[] shapes)
    {
        if (shapes.Distinct().Count() != 1)
        {
            throw new InvalidShapeException($"The shapes of the tensors are different but should be the same. {string.Join(", ", shapes.Select(x => x.ToString()).ToArray())}");
        }
    }

    /// <summary>
    /// Throws an exception if the two shapes have different element counts.
    /// </summary>
    /// <param name="shapes">The shapes to compare.</param>
    /// <exception cref="InvalidShapeException">Throws when the element counts are different.</exception>
    [StackTraceHidden]
    public static void ThrowIfDifferentElementCount(params Shape[] shapes)
    {
        if (shapes.Distinct().Count() != 1)
        {
            throw new InvalidShapeException($"The shapes of the tensors have different element counts but should be the same. {string.Join(", ", shapes.Select(x => x.ToString()).ToArray())}");
        }
    }
}