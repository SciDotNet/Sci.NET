// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Diagnostics;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// Represents the shape of an N-dimensional array.
/// </summary>
[PublicAPI]
[DebuggerTypeProxy(typeof(Array))]
[DebuggerDisplay("{DebuggerProxy}")]
public sealed class Shape : IEnumerable<int>, IEquatable<Shape>, IFormattable
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">The dimensions of the tensorShape.</param>
    public Shape(params int[] dimensions)
    {
        if (dimensions.LongLength == 0)
        {
            Dimensions = Array.Empty<int>();
            Strides = Array.Empty<long>();
            ElementCount = 1;
            return;
        }

        Dimensions = dimensions;
        Strides = new long[Rank];
        Strides[Rank - 1] = 1;
        ElementCount = 1;

        for (var i = 0; i < Rank; i++)
        {
            ElementCount *= Dimensions[i];
        }

        for (var i = Rank - 2; i >= 0; i--)
        {
            Strides[i] = Strides[i + 1] * Dimensions[i + 1];
        }
    }

    /// <summary>
    /// Gets the size of each dimension.
    /// </summary>
    public int[] Dimensions { get; }

    /// <summary>
    /// Gets the stride of each dimension.
    /// </summary>
    public long[] Strides { get; }

    /// <summary>
    /// Gets the rank of the tensor.
    /// </summary>
    public int Rank => Dimensions.Length;

    /// <summary>
    /// Gets the total number of elements in the tensor.
    /// </summary>
    public long ElementCount { get; }

    /// <summary>
    /// Gets a value indicating whether the tensorShape is a scalar.
    /// </summary>
    public bool IsScalar => ElementCount == 1;

    /// <summary>
    /// Gets a value indicating whether the tensorShape is a vector.
    /// </summary>
    public bool IsVector => ElementCount > 1 && Rank == 1;

    /// <summary>
    /// Gets a value indicating whether the tensorShape is a matrix.
    /// </summary>
    public bool IsMatrix => ElementCount > 1 && Rank == 2;

    [DebuggerHidden] private string DebuggerProxy => string.Join(',', Dimensions);

    /// <summary>
    /// Gets the dimension at the given index.
    /// </summary>
    /// <param name="index">The index to find the dimension of.</param>
    public long this[Index index] => Dimensions[index];

    /// <summary>
    /// Equal to operator overload.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>A value indicating whether the left and right operands are equal.</returns>
    public static bool operator ==(Shape left, Shape right)
    {
        return left.Equals(right);
    }

    /// <summary>
    /// Not equal to operator overload.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>A value indicating whether the left and right operands are equal.</returns>
    public static bool operator !=(Shape left, Shape right)
    {
        return !left.Equals(right);
    }

    /// <summary>
    /// Gets the index of the element at the given indices.
    /// </summary>
    /// <param name="indices">The array of indices to get the linear index for.</param>
    /// <returns>The linear index for the given N-Dimension index.</returns>
    /// <exception cref="ArgumentException">The indices array length was not equal to the number of dimensions.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The index of a given dimension was outside the bounds of the dimension.</exception>
    public long GetLinearIndex(params int[] indices)
    {
        if (indices.LongLength != Rank)
        {
            throw new ArgumentException("The number of indices must match the rank of the tensor.");
        }

        var linearIndex = 0L;

        for (var i = 0; i < Rank; i++)
        {
            if (indices[i] < 0 || indices[i] >= Dimensions[i])
            {
                throw new ArgumentOutOfRangeException(
                    nameof(indices),
                    "The indices must be within the bounds of the tensor.");
            }

            linearIndex += indices[i] * Strides[i];
        }

        return linearIndex;
    }

    /// <summary>
    /// Gets the multi dimensional indices of the element at the given linear index.
    /// </summary>
    /// <param name="linearIndex">The linear index.</param>
    /// <returns>The multi dimensional indices of the element at the given linear index.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The given index was out of the range of acceptable values.</exception>
    public int[] GetIndicesFromLinearIndex(long linearIndex)
    {
        if (linearIndex >= ElementCount || linearIndex < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(linearIndex),
                "The linear index must be within the bounds of the tensor.");
        }

        var index = new int[Rank];

        for (var j = Rank - 1; j >= 0; j--)
        {
            index[j] = (int)(linearIndex % this[j]);
            linearIndex /= this[j];
        }

        return index;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(Shape? other)
    {
        return other is not null && Dimensions.SequenceEqual(other.Dimensions);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is Shape other && Equals(other);
    }

    /// <inheritdoc />
    public IEnumerator<int> GetEnumerator()
    {
        return (IEnumerator<int>)Dimensions.GetEnumerator();
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        var result = 0;
        var shift = 0;
        foreach (var dim in Dimensions)
        {
            shift = (shift + 11) % 21;
            result ^= (dim + 1024) << shift;
        }

        return result;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"Shape<{string.Join(", ", Dimensions)}>";
    }

    /// <inheritdoc />
    public string ToString(string? format, IFormatProvider? formatProvider)
    {
        return ToString();
    }

    /// <inheritdoc/>
    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}