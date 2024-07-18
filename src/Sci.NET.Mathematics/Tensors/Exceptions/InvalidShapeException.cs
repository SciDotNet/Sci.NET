// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

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
}