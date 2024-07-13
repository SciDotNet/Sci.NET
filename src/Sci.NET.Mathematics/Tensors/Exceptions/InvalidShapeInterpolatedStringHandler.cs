// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Text;

namespace Sci.NET.Mathematics.Tensors.Exceptions;

/// <summary>
/// The interpolated string handler for <see cref="InvalidShapeException"/>.
/// </summary>
[InterpolatedStringHandler]
[ExcludeFromCodeCoverage]
[PublicAPI]
public readonly ref struct InvalidShapeInterpolatedStringHandler
{
    private readonly StringBuilder _builder;
    private readonly List<Shape> _shapes;

    /// <summary>
    /// Initializes a new instance of the <see cref="InvalidShapeInterpolatedStringHandler"/> struct.
    /// </summary>
    /// <param name="literalLength">The length of the literal.</param>
    /// <param name="formattedCount">The number of formatted values.</param>
    public InvalidShapeInterpolatedStringHandler(int literalLength, int formattedCount)
    {
        _builder = new StringBuilder(literalLength + (formattedCount * 4));
        _shapes = new List<Shape>(formattedCount);
    }

    /// <summary>
    /// Appends the given <paramref name="literal"/> to the formatted string.
    /// </summary>
    /// <param name="literal">The literal to append.</param>
    public void AppendLiteral(string literal)
    {
        _ = _builder.Append(literal);
    }

    /// <summary>
    /// Appends the given <paramref name="value"/> to the formatted string.
    /// </summary>
    /// <typeparam name="T">The type of the value to append.</typeparam>
    /// <param name="value">The value to append.</param>
    public void AppendFormatted<T>(T value)
    {
        if (value is Shape shape)
        {
            _shapes.Add(shape);
        }

        _ = _builder.Append(value);
    }

    /// <summary>
    /// Gets the shapes used in the formatted string.
    /// </summary>
    /// <returns>The shapes used in the formatted string.</returns>
    public IReadOnlyCollection<Shape> GetShapes()
    {
        var shapes = new Shape[_shapes.Count];
        _shapes.CopyTo(shapes);
        return shapes;
    }

    internal string GetFormattedString()
    {
        return _builder.ToString();
    }
}