﻿// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Sci.NET.Common.Performance;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// Represents the shape of an N-dimensional array.
/// </summary>
[PublicAPI]
public sealed class Shape : IEnumerable<int>, IEquatable<Shape>, IFormattable
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">The dimensions of the shape.</param>
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

        for (var i = Rank - 1; i >= 0; i--)
        {
            Strides[i] = ElementCount;
            ElementCount *= dimensions[i];
        }
    }

    private Shape(int[] dimensions, long dataOffset)
        : this(dimensions)
    {
        DataOffset = dataOffset;
    }

    /// <summary>
    /// Gets the dimensions of the <see cref="Shape"/>.
    /// </summary>
    public int[] Dimensions { get; }

    /// <summary>
    /// Gets the strides of the <see cref="Shape"/>.
    /// </summary>
    public long[] Strides { get; }

    /// <summary>
    /// Gets the rank of the <see cref="Shape"/>.
    /// </summary>
    public int Rank => Dimensions.Length;

    /// <summary>
    /// Gets a value indicating whether the <see cref="Shape"/> is a scalar.
    /// </summary>
    public bool IsScalar => Rank == 0;

    /// <summary>
    /// Gets a value indicating whether the <see cref="Shape"/> is a vector.
    /// </summary>
    public bool IsVector => Rank == 1;

    /// <summary>
    /// Gets a value indicating whether the <see cref="Shape"/> is a matrix.
    /// </summary>
    public bool IsMatrix => Rank == 2;

    /// <summary>
    /// Gets a value indicating whether the <see cref="Shape"/> is a tensor.
    /// </summary>
    public bool IsTensor => Rank > 2;

    /// <summary>
    /// Gets the number of elements in the <see cref="Shape"/>.
    /// </summary>
    public long ElementCount { get; }

    /// <summary>
    /// Gets the offset of the data in the underlying storage.
    /// </summary>
    public long DataOffset { get; }

    /// <summary>
    /// Gets the length of the dimension at the specified <paramref name="index"/>.
    /// </summary>
    /// <param name="index">The index of the dimension to query.</param>
    public int this[Index index]
    {
        get
        {
            if (Rank == 0 && index.GetOffset(1) == 0)
            {
                return 1;
            }

            return Dimensions[index.GetOffset(Rank)];
        }
    }

    /// <summary>
    /// Gets the range of dimensions at the specified <paramref name="range"/>.
    /// </summary>
    /// <param name="range">The range of dimensions to query.</param>
    public int[] this[Range range] => Dimensions[range];

    /// <summary>
    /// Equals operator.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>A value indicating whether the <paramref name="left"/> and
    /// <paramref name="right"/> operands are equal.</returns>
    public static bool operator ==(Shape left, Shape right)
    {
        return left.Equals(right);
    }

    /// <summary>
    /// Not equals operator.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>A value indicating whether the <paramref name="left"/> and
    /// <paramref name="right"/> operands are not equal.</returns>
    public static bool operator !=(Shape left, Shape right)
    {
        return !left.Equals(right);
    }

    /// <summary>
    /// Gets a new scalar <see cref="Shape"/>.
    /// </summary>
    /// <returns>A new scalar <see cref="Shape"/>.</returns>
    public static Shape Scalar()
    {
        return new();
    }

    /// <summary>
    /// Creates a new <see cref="Shape"/> with the given length.
    /// </summary>
    /// <param name="length">A vector with the given length.</param>
    /// <returns>A shape representing a vector of the given <paramref name="length"/>.</returns>
    public static Shape Vector(int length)
    {
        return new(length);
    }

    /// <summary>
    /// Creates a new <see cref="Shape"/> with the given dimensions.
    /// </summary>
    /// <param name="rows">The number of rows.</param>
    /// <param name="columns">The number of columns.</param>
    /// <returns>The new <see cref="Shape"/>.</returns>
    public static Shape Matrix(int rows, int columns)
    {
        return new(rows, columns);
    }

    /// <summary>
    /// Creates a new <see cref="Shape"/> with the given dimensions.
    /// </summary>
    /// <param name="dimensions">The dimensions of the tensor.</param>
    /// <returns>The new <see cref="Shape"/>.</returns>
    public static Shape Tensor(params int[] dimensions)
    {
        return new(dimensions);
    }

    /// <summary>
    /// Creates a new <see cref="Shape"/> with the given dimensions.
    /// </summary>
    /// <param name="axes">The axes to expand the dimensions by.</param>
    /// <returns>The new <see cref="Shape"/>.</returns>
    public Shape ExpandDims(int[] axes)
    {
        var sortedAxes = axes.Order().ToArray();
        var newShape = new List<int>(Dimensions);

        foreach (var axis in sortedAxes)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(axis, 0);
            ArgumentOutOfRangeException.ThrowIfGreaterThan(axis, newShape.Count);

            newShape.Insert(axis, 1);
        }

        return new Shape(newShape.ToArray(), DataOffset);
    }

    /// <summary>
    /// Pads the shape with leading dimensions of 1 to the given rank.
    /// </summary>
    /// <param name="rank">The rank to pad the shape to.</param>
    /// <returns>The new <see cref="Shape"/>.</returns>
    public Shape PadShape(int rank)
    {
        var newShape = new List<int>(Dimensions);

        while (newShape.Count < rank)
        {
            newShape.Insert(0, 1);
        }

        return new Shape(newShape.ToArray(), DataOffset);
    }

    /// <summary>
    /// Gets the multi dimensional indices of the element at the given linear index.
    /// </summary>
    /// <param name="linearIndex">The linear index.</param>
    /// <returns>The multi dimensional indices of the element at the given linear index.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The given index was out of the range of acceptable values.</exception>
    [MethodImpl(ImplementationOptions.HotPath)]
    public int[] GetIndicesFromLinearIndex(long linearIndex)
    {
        linearIndex -= DataOffset;

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

    /// <summary>
    /// Gets the index of the element at the given indices.
    /// </summary>
    /// <param name="indices">The array of indices to get the linear index for.</param>
    /// <returns>The linear index for the given N-Dimension index.</returns>
    /// <exception cref="ArgumentException">The indices array length was not equal to the number of dimensions.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The index of a given dimension was outside the bounds of the dimension.</exception>
    [MethodImpl(ImplementationOptions.HotPath)]
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

        return linearIndex + DataOffset;
    }

    /// <summary>
    /// Creates a slice of the current <see cref="Shape"/>.
    /// </summary>
    /// <param name="axes">The start index of the slice.</param>
    /// <returns>The new <see cref="Shape"/>.</returns>
    /// <exception cref="ArgumentException">The slice indices were invalid.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The index was out of range for the axis.</exception>
    /// <exception cref="InvalidOperationException">The output shape could not be calculated.</exception>
    public Shape Slice(params int[] axes)
    {
        if (axes.Length > Rank)
        {
            throw new ArgumentException("The number of slice indices must be less than or equal to the rank of the shape.");
        }

        if (axes.Length == Rank)
        {
            return new Shape(Array.Empty<int>(), GetLinearIndex(axes));
        }

        for (var i = 0; i < axes.Length; i++)
        {
            if (axes[i] < 0 || axes[i] >= Dimensions[i])
            {
                throw new ArgumentOutOfRangeException($"The index {axes[i]} is out of range for axis {i}.");
            }
        }

        var newRank = Rank - axes.Length;
        var newDimensions = new int[newRank];
        var dataOffset = DataOffset;

        for (var i = newRank - 1; i >= 0; i--)
        {
            newDimensions[i] = Dimensions[i + axes.Length];
        }

        for (var i = 0; i < axes.Length; i++)
        {
            dataOffset += axes[i] * Strides[i];
        }

        var newDataOffset = DataOffset + dataOffset;

        return new Shape(newDimensions, newDataOffset);
    }

    /// <summary>
    /// Determines whether the <paramref name="other"/> instance is equal to the current <see cref="Shape"/> instance.
    /// </summary>
    /// <param name="other">The instance to compare this instance to.</param>
    /// <returns>A value indicating whether the current instance is equal to the <paramref name="other"/> instance.</returns>
    public bool Equals(Shape? other)
    {
        if (other is null)
        {
            return false;
        }

        if (other.Rank != Rank ||
            other.ElementCount != ElementCount ||
            other.DataOffset != DataOffset)
        {
            return false;
        }

        for (var i = 0; i < Rank; i++)
        {
            if (Dimensions[i] != other.Dimensions[i])
            {
                return false;
            }
        }

        return true;
    }

    /// <inheritdoc />
    public override bool Equals(object? obj)
    {
        return obj is Shape shape && Equals(shape);
    }

    /// <inheritdoc />
    public IEnumerator<int> GetEnumerator()
    {
        return Dimensions.LongLength == 0
            ? Enumerable.Empty<int>().GetEnumerator()
            : ((IEnumerable<int>)Dimensions).GetEnumerator();
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return ToString(null, null);
    }

    /// <inheritdoc />
    public string ToString(string? format, IFormatProvider? formatProvider)
    {
        return $"[{string.Join(", ", Dimensions)}]";
    }

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var result = 0;

        for (var index = 0; index < Dimensions.Length; index++)
        {
            var dim = Dimensions[index];
            var stride = Strides[index];
            result = HashCode.Combine(result, stride, dim);
        }

        return HashCode.Combine(result, DataOffset);
    }

    /// <inheritdoc/>
    [ExcludeFromCodeCoverage]
    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}