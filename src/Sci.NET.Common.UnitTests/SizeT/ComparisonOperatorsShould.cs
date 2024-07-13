// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.UnitTests.SizeT;

public class ComparisonOperatorsShould
{
    [Theory]
    [InlineData(0, 0, true)]
    [InlineData(0, 1, false)]
    [InlineData(1, 0, false)]
    [InlineData(1, 1, true)]
    public void EqualsOperator_ReturnsExpectedResult(int left, int right, bool expected)
    {
        var leftSizeT = new Common.SizeT(left);
        var rightSizeT = new Common.SizeT(right);

        var result = leftSizeT == rightSizeT;

        result.Should().Be(expected);
    }

    [Theory]
    [InlineData(0, 0, false)]
    [InlineData(0, 1, true)]
    [InlineData(1, 0, true)]
    [InlineData(1, 1, false)]
    public void NotEqualsOperator_ReturnsExpectedResult(int left, int right, bool expected)
    {
        var leftSizeT = new Common.SizeT(left);
        var rightSizeT = new Common.SizeT(right);

        var result = leftSizeT != rightSizeT;

        result.Should().Be(expected);
    }

    [Theory]
    [InlineData(0, 0, false)]
    [InlineData(0, 1, false)]
    [InlineData(1, 0, true)]
    [InlineData(1, 1, false)]
    public void GreaterThanOperator_ReturnsExpectedResult(int left, int right, bool expected)
    {
        var leftSizeT = new Common.SizeT(left);
        var rightSizeT = new Common.SizeT(right);

        var result = leftSizeT > rightSizeT;

        result.Should().Be(expected);
    }

    [Theory]
    [InlineData(0, 0, false)]
    [InlineData(0, 1, true)]
    [InlineData(1, 0, false)]
    [InlineData(1, 1, false)]
    public void LessThanOperator_ReturnsExpectedResult(int left, int right, bool expected)
    {
        var leftSizeT = new Common.SizeT(left);
        var rightSizeT = new Common.SizeT(right);

        var result = leftSizeT < rightSizeT;

        result.Should().Be(expected);
    }
}