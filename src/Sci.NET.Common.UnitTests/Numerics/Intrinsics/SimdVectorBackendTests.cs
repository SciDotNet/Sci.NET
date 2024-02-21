// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Numerics.Intrinsics;

namespace Sci.NET.Common.UnitTests.Numerics.Intrinsics;

public class SimdVectorBackendTests
{
    [Fact]
    public unsafe void Ctor_ReturnsNewInstance_GivenPointer()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        var values = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            values[i] = i;
        }

        // Act
        var result = new SimdVectorBackend<float>(values);

        // Assert
        for (var i = 0; i < count; i++)
        {
            result[i].Should().Be(i);
        }
    }

    [Fact]
    public void Ctor_ReturnsNewInstance_GivenSpan()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> span = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            span[i] = i;
        }

        // Act
        var result = new SimdVectorBackend<float>(span);

        // Assert
        for (var i = 0; i < count; i++)
        {
            result[i].Should().Be(i);
        }
    }

    [Fact]
    public void Ctor_ReturnsNewInstance_GivenVector()
    {
        // Arrange
        const float value = 2.0f;
        var vector = new Vector<float>(value);

        // Act
        var result = new SimdVectorBackend<float>(vector);

        // Assert
        for (var i = 0; i < result.Count; i++)
        {
            result[i].Should().Be(value);
        }
    }

    [Fact]
    public void EqualityMembers_ReturnsExpectedResult_GivenEqualValues()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> leftValues = stackalloc float[count];
        Span<float> rightValues = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            leftValues[i] = i;
            rightValues[i] = i;
        }

        var rightVector = new SimdVectorBackend<float>(leftValues);
        var leftVector = new SimdVectorBackend<float>(rightValues);

        // Act
        var equalsOperatorResult = leftVector == rightVector;
        var notEqualsOperatorResult = leftVector != rightVector;
        var methodResult = leftVector.Equals(rightVector);

        // Assert
        equalsOperatorResult.Should().BeTrue();
        notEqualsOperatorResult.Should().BeFalse();
        methodResult.Should().BeTrue();
    }

    [Fact]
    public void EqualityMembers_ReturnsExpectedResult_GivenNonEqualValues()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> leftValues = stackalloc float[count];
        Span<float> rightValues = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            leftValues[i] = i;
            rightValues[i] = 2.0f + i;
        }

        var rightVector = new SimdVectorBackend<float>(leftValues);
        var leftVector = new SimdVectorBackend<float>(rightValues);

        // Act
        var equalsOperatorResult = leftVector == rightVector;
        var notEqualsOperatorResult = leftVector != rightVector;
        var methodResult = leftVector.Equals(rightVector);

        // Assert
        equalsOperatorResult.Should().BeFalse();
        notEqualsOperatorResult.Should().BeTrue();
        methodResult.Should().BeFalse();
    }

    [Fact]
    public void Add_ReturnsCorrectSum()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> left = stackalloc float[count];
        Span<float> right = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            left[i] = i;
            right[i] = i;
        }

        var leftVector = new SimdVectorBackend<float>(left);
        var rightVector = new SimdVectorBackend<float>(right);

        // Act
        var result = leftVector.Add(rightVector);

        // Assert
        for (var i = 0; i < count; i++)
        {
            result[i].Should().Be(left[i] + right[i]);
        }
    }

    [Fact]
    public void GetHashCode_ReturnsExpectedResult()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> values = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            values[i] = i;
        }

        var vector = new SimdVectorBackend<float>(values);

        // Act
        var result = vector.GetHashCode();

        // Assert
        result.Should().Be(vector.GetHashCode());
    }

    [Fact]
    public void Subtract_ReturnsCorrectDifference()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> left = stackalloc float[count];
        Span<float> right = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            left[i] = i * 2;
            right[i] = i;
        }

        var leftVector = new SimdVectorBackend<float>(left);
        var rightVector = new SimdVectorBackend<float>(right);

        // Act
        var result = leftVector.Subtract(rightVector);

        // Assert
        for (var i = 0; i < count; i++)
        {
            result[i].Should().Be(left[i] - right[i]);
        }
    }

    [Fact]
    public void Multiply_ReturnsCorrectProduct()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> left = stackalloc float[count];
        Span<float> right = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            left[i] = i;
            right[i] = i;
        }

        var leftVector = new SimdVectorBackend<float>(left);
        var rightVector = new SimdVectorBackend<float>(right);

        // Act
        var result = leftVector.Multiply(rightVector);

        // Assert
        for (var i = 0; i < count; i++)
        {
            result[i].Should().Be(left[i] * right[i]);
        }
    }

    [Fact]
    public void Divide_ReturnsCorrectQuotient()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> left = stackalloc float[count];
        Span<float> right = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            left[i] = i * 2;
            right[i] = i + 1;
        }

        var leftVector = new SimdVectorBackend<float>(left);
        var rightVector = new SimdVectorBackend<float>(right);

        // Act
        var result = leftVector.Divide(rightVector);

        // Assert
        for (var i = 0; i < count; i++)
        {
            result[i].Should().Be(left[i] / right[i]);
        }
    }

    [Fact]
    public void Sqrt_ReturnsCorrectResult()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> values = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            values[i] = (i + 2) * (i + 2);
        }

        var vector = new SimdVectorBackend<float>(values);

        // Act
        var result = vector.Sqrt();

        // Assert
        for (var i = 0; i < count; i++)
        {
            result[i].Should().Be(MathF.Sqrt(values[i]));
        }
    }

    [Fact]
    public void Abs_ReturnsCorrectResult()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> values = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            values[i] = -i;
        }

        var vector = new SimdVectorBackend<float>(values);

        // Act
        var result = vector.Abs();

        // Assert
        for (var i = 0; i < count; i++)
        {
            result[i].Should().Be(MathF.Abs(values[i]));
        }
    }

    [Fact]
    public void Negate_ReturnsCorrectResult()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> values = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            values[i] = i;
        }

        var vector = new SimdVectorBackend<float>(values);

        // Act
        var result = vector.Negate();

        // Assert
        for (var i = 0; i < count; i++)
        {
            result[i].Should().Be(-values[i]);
        }
    }

    [Fact]
    public void Max_ReturnsCorrectResult()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> left = stackalloc float[count];
        Span<float> right = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            left[i] = i;
            right[i] = i + 1;
        }

        var leftVector = new SimdVectorBackend<float>(left);
        var rightVector = new SimdVectorBackend<float>(right);

        // Act
        var result = leftVector.Max(rightVector);

        // Assert
        for (var i = 0; i < count; i++)
        {
            result[i].Should().Be(MathF.Max(left[i], right[i]));
        }
    }

    [Fact]
    public void Min_ReturnsCorrectResult()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> left = stackalloc float[count];
        Span<float> right = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            left[i] = i;
            right[i] = i + 1;
        }

        var leftVector = new SimdVectorBackend<float>(left);
        var rightVector = new SimdVectorBackend<float>(right);

        // Act
        var result = leftVector.Min(rightVector);

        // Assert
        for (var i = 0; i < count; i++)
        {
            result[i].Should().Be(MathF.Min(left[i], right[i]));
        }
    }

    [Fact]
    public void Clamp_ReturnsCorrectResult()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> values = stackalloc float[count];
        Span<float> minValues = stackalloc float[count];
        Span<float> maxValues = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            values[i] = i;
            minValues[i] = i - 1;
            maxValues[i] = i + 1;
        }

        var vector = new SimdVectorBackend<float>(values);
        var minVector = new SimdVectorBackend<float>(minValues);
        var maxVector = new SimdVectorBackend<float>(maxValues);

        // Act
        var result = vector.Clamp(minVector, maxVector);

        // Assert
        for (var i = 0; i < count; i++)
        {
            result[i].Should().Be(Math.Clamp(values[i], minValues[i], maxValues[i]));
        }
    }

    [Fact]
    public void Dot_ReturnsCorrectResult()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> left = stackalloc float[count];
        Span<float> right = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            left[i] = i;
            right[i] = i;
        }

        var leftVector = new SimdVectorBackend<float>(left);
        var rightVector = new SimdVectorBackend<float>(right);

        // Act
        var result = leftVector.Dot(rightVector);

        // Assert
        var expected = 0.0f;

        for (var i = 0; i < count; i++)
        {
            expected += left[i] * right[i];
        }

        result.Should().Be(expected);
    }

    [Fact]
    public void Sum_ReturnsCorrectResult()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> values = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            values[i] = i;
        }

        var vector = new SimdVectorBackend<float>(values);

        // Act
        var result = vector.Sum();

        // Assert
        var expected = 0.0f;

        for (var i = 0; i < count; i++)
        {
            expected += values[i];
        }

        result.Should().Be(expected);
    }

    [Fact]
    public void SquareDifference_ReturnsCorrectResult()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> left = stackalloc float[count];
        Span<float> right = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            left[i] = i * 2;
            right[i] = i;
        }

        var leftVector = new SimdVectorBackend<float>(left);
        var rightVector = new SimdVectorBackend<float>(right);

        // Act
        var result = leftVector.SquareDifference(rightVector);

        // Assert
        for (var i = 0; i < count; i++)
        {
            result[i].Should().Be((left[i] - right[i]) * (left[i] - right[i]));
        }
    }

    [Fact]
    public void CreateDuplicateZeroed_ReturnsNewInstance()
    {
        // Arrange
        var vector = new SimdVectorBackend<float>(new float[SimdVector.Count<float>()]);

        // Act
        var result = vector.CreateDuplicateZeroed();

        // Assert
        result.Should().BeOfType<SimdVectorBackend<float>>();

        for (var i = 0; i < result.Count; i++)
        {
            result[i].Should().Be(0.0f);
        }
    }

    [Fact]
    public void CopyTo_CopiesValuesToSpan()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> values = stackalloc float[count];

        for (var i = 0; i < count; i++)
        {
            values[i] = i;
        }

        var vector = new SimdVectorBackend<float>(values);
        var span = new float[count];

        // Act
        vector.CopyTo(span);

        // Assert
        for (var i = 0; i < count; i++)
        {
            span[i].Should().Be(i);
        }
    }

    [Fact]
    public void CreateWith_ReturnsNewInstance_GivenSpan()
    {
        // Arrange
        var count = SimdVector.Count<float>();
        Span<float> values = stackalloc float[count];
        var original = default(SimdVectorBackend<float>);

        for (var i = 0; i < count; i++)
        {
            values[i] = i;
        }

        // Act
        var result = original.CreateWith(values);

        // Assert
        for (var i = 0; i < count; i++)
        {
            result[i].Should().Be(i);
        }
    }
}