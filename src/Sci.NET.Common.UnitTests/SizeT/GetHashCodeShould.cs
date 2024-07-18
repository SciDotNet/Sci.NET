// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.UnitTests.SizeT;

public class GetHashCodeShould
{
    [Theory]
    [InlineData(0, 0, true)]
    [InlineData(0, 1, false)]
    [InlineData(1, 0, false)]
    [InlineData(1, 1, true)]
    public void ReturnSameHashCode_GivenEqualValues(int left, int right, bool areEqual)
    {
        var leftSizeT = new Common.SizeT(left);
        var rightSizeT = new Common.SizeT(right);

        var leftHashCode = leftSizeT.GetHashCode();
        var rightHashCode = rightSizeT.GetHashCode();

        if (areEqual)
        {
            leftHashCode.Should().Be(rightHashCode);
        }
        else
        {
            leftHashCode.Should().NotBe(rightHashCode);
        }
    }
}