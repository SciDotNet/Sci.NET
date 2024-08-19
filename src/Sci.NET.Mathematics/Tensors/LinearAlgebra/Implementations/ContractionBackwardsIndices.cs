// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Comparison;

namespace Sci.NET.Mathematics.Tensors.LinearAlgebra.Implementations;

/// <summary>
/// Represents the indices of the contracted dimensions of a tensor.
/// </summary>
internal readonly struct ContractionBackwardsIndices : IValueEquatable<ContractionBackwardsIndices>
{
    public required (int[] LeftIndices, int[] RightIndices) LeftGradIndices { get; init; }

    public required (int[] LeftIndices, int[] RightIndices) RightGradIndices { get; init; }

    public static bool operator ==(ContractionBackwardsIndices left, ContractionBackwardsIndices right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(ContractionBackwardsIndices left, ContractionBackwardsIndices right)
    {
        return !left.Equals(right);
    }

    public bool Equals(ContractionBackwardsIndices other)
    {
        return LeftGradIndices.Equals(other.LeftGradIndices) && RightGradIndices.Equals(other.RightGradIndices);
    }

    public override bool Equals(object? obj)
    {
        return obj is ContractionBackwardsIndices other && Equals(other);
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(LeftGradIndices, RightGradIndices);
    }
}