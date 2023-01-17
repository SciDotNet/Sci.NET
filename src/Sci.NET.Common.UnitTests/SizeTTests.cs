// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.UnitTests;

public class SizeTTests
{
    [Theory]
    [InlineData(0, 0, true)]
    [InlineData(0, 1, false)]
    [InlineData(1, 0, false)]
    [InlineData(1, 1, true)]
    public void Equals_WhenCalled_ReturnsExpectedResult(int first, int second, bool expectedResult)
    {
        var size1 = new SizeT(first);
        var size2 = new SizeT(second);

        var result1 = size1.Equals(size2);
        var result2 = size1 == size2;

        result1.Should()
            .Be(expectedResult);
        result2.Should()
            .Be(expectedResult);
    }

    [Theory]
    [InlineData(0, 0, false)]
    [InlineData(0, 1, true)]
    [InlineData(1, 0, true)]
    [InlineData(1, 1, false)]
    public void NotEquals_WhenCalled_ReturnsExpectedResult(int first, int second, bool expectedResult)
    {
        var size1 = new SizeT(first);
        var size2 = new SizeT(second);

        var result = size1 != size2;

        result.Should()
            .Be(expectedResult);
    }

    [Theory]
    [InlineData(0, 0, true)]
    [InlineData(0, 1, false)]
    [InlineData(1, 0, false)]
    [InlineData(1, 1, true)]
    public void Equals_GivenInt32_ReturnsExpectedResult(int left, int right, bool expectation)
    {
        var size = new SizeT(left);

        var result = size.Equals(right);

        result.Should()
            .Be(expectation);
    }

    [Theory]
    [InlineData(0, 0, true)]
    [InlineData(0, 1, false)]
    [InlineData(100, 101, false)]
    [InlineData(100, 100, true)]
    public void GetHashCode_WhenCalled_ReturnsExpectedResult(int first, int second, bool shouldBeEqual)
    {
        var size1 = new SizeT(first);
        var size2 = new SizeT(second);

        var result = size1.GetHashCode() == size2.GetHashCode();

        result.Should()
            .Be(shouldBeEqual);
    }

    [Fact]
    public void Equals_GivenObject_ReturnsFalse()
    {
        var size = new SizeT(0);

        var result = size.Equals(new object());

        result.Should()
            .BeFalse();
    }

    [Fact]
    public unsafe void Ctor_GivenIntPtr_ReturnsExpectedResult()
    {
        var size = new SizeT(new nint(100));

        var value = (long*)&size;

        (*value).Should()
            .Be(100);
    }

    [Fact]
    public unsafe void Ctor_GivenUIntPtr_ReturnsExpectedResult()
    {
        var size = new SizeT(new nuint(100));

        var value = (long*)&size;

        (*value).Should()
            .Be(100);
    }

    [Fact]
    public unsafe void Ctor_GivenUInt_ReturnsExpectedResult()
    {
        var size = new SizeT(100U);

        var value = (long*)&size;

        (*value).Should()
            .Be(100);
    }

    [Fact]
    public unsafe void Ctor_GivenLong_ReturnsExpectedResult()
    {
        var size = new SizeT(100L);

        var value = (long*)&size;

        (*value).Should()
            .Be(100);
    }

    [Fact]
    public unsafe void Ctor_GivenULong_ReturnsExpectedResult()
    {
        var size = new SizeT(100UL);

        var value = (long*)&size;

        (*value).Should()
            .Be(100);
    }

    [Fact]
    public void Zero_EqualsZero_ReturnsTrue()
    {
        var size1 = SizeT.Zero;
        var size2 = new SizeT(0);

        var result = size1.Equals(size2);

        result.Should()
            .BeTrue();
    }

    [Theory]
    [InlineData(100)]
    [InlineData(0)]
    [InlineData(long.MaxValue)]
    public unsafe void FromInt64_GivenSizeT_ReturnsExpectedResult(long value)
    {
        SizeT sizeT1 = value;
        var sizeT2 = SizeT.FromInt64(value);

        var result1 = (long*)&sizeT1;
        var result2 = (long*)&sizeT2;

        (*result1).Should()
            .Be(value);
        (*result2).Should()
            .Be(value);
    }

    [Theory]
    [InlineData(100)]
    [InlineData(0)]
    [InlineData(long.MaxValue)]
    public unsafe void ToInt64_GivenSizeT_ReturnsExpectedResult(long value)
    {
        var sizeT = *(SizeT*)&value;

        var result = sizeT.ToInt64();

        result.Should()
            .Be(value);
    }

    [Theory]
    [InlineData(100)]
    [InlineData(0)]
    [InlineData(ulong.MaxValue)]
    public unsafe void ToUIntPtr_GivenSizeT_ReturnsExpectedResult(ulong value)
    {
        var sizeT1 = *(SizeT*)&value;

        var result = sizeT1.ToUIntPtr();

        result.Should()
            .Be(new nuint(value));
    }

    [Theory]
    [InlineData(100)]
    [InlineData(0)]
    [InlineData(ulong.MaxValue)]
    public unsafe void ToIntPtr_GivenSizeT_ReturnsExpectedResult(ulong value)
    {
        var sizeT1 = *(SizeT*)&value;

        var result = sizeT1.ToIntPtr();

        result.Should()
            .Be(new nint((long)value));
    }
}