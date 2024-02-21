// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Numerics.Intrinsics;

namespace Sci.NET.Common.UnitTests.Numerics.Intrinsics;

public class SimdVectorTests
{
    [Fact]
    public void Count_IsGreaterThanOne_GivenFloat()
    {
        // Assumes that most CPUs will support fp32 vectors.
        SimdVector.Count<float>().Should().BeGreaterThan(1);
    }

    [Fact]
    public void Count_ReturnsOne_GivenDouble()
    {
        SimdVector.Count<double>().Should().Be(1);
    }

    [Fact]
    public void Create_ReturnsZeroedVector_GivenFloatType()
    {
        // Assumes most CPUs will support float vectors
        var result = SimdVector.Create<float>();

        result.Should().BeOfType<SimdVectorBackend<float>>();

        for (var i = 0; i < result.Count; i++)
        {
            result[i].Should().Be(0);
        }
    }

    [Fact]
    public void Create_ReturnsZeroedVector_GivenFloatTypeWithValue()
    {
        // Assumes most CPUs will support float vectors
        const float value = 2.0f;
        var result = SimdVector.Create(value);

        result.Should().BeOfType<SimdVectorBackend<float>>();

        for (var i = 0; i < result.Count; i++)
        {
            result[i].Should().Be(value);
        }
    }

    [Fact]
    public void Create_ReturnsZeroedScalar_GivenDoubleWithValue()
    {
        const double value = 2.0d;
        var result = SimdVector.Create(value);

        result.Should().BeOfType<SimdScalarBackend<double>>();
        result[0].Should().Be(value);
    }

    [Fact]
    public void Create_ReturnsZeroedScalar_GivenDoubleTypeAndNumber()
    {
        var result = SimdVector.Create<double>();

        result.Should().BeOfType<SimdScalarBackend<double>>();
        result[0].Should().Be(0);
    }
}