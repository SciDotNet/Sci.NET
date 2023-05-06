// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.Tensors.Exceptions;

/// <summary>
/// The exception that is thrown when a <see cref="ITensor{TNumber}"/> has an invalid shape.
/// </summary>
[PublicAPI]
public class InvalidShapeException : Exception
{
    /// <summary>
    /// Initializes a new instance of the <see cref="InvalidShapeException"/> class.
    /// </summary>
    /// <param name="reason">The reason the shape is invalid.</param>
    /// <param name="shapes">The shapes causing the exception.</param>
    [StringFormatMethod(nameof(reason))]
    public InvalidShapeException(string reason, params Shape[] shapes)
#pragma warning disable CA1305
        : base($"The given shape is invalid. {string.Format(reason, shapes.GetEnumerator())}.")
#pragma warning restore CA1305
    {
        GivenShape = shapes;
    }

    /// <summary>
    /// Gets the given shape.
    /// </summary>
    public IEnumerable<Shape> GivenShape { get; }
}