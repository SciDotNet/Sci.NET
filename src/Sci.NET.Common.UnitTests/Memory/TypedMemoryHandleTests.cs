// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory;

public class TypedMemoryHandleTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public unsafe void ToPointer_WhenCalled_ReturnsExpectedResult(int intPointer)
    {
        var handle = new TypedMemoryHandle<int>((int*)intPointer);

        var result = handle.ToPointer();

        ((int)result).Should().Be(intPointer);
    }

    [Theory]
    [InlineData(0, 0, true)]
    [InlineData(0, 1, false)]
    [InlineData(1, 0, false)]
    [InlineData(1, 1, true)]
    public unsafe void Equals_WhenCalled_ReturnsExpectedResult(int intPointer1, int intPointer2, bool expectedResult)
    {
        var handle1 = new TypedMemoryHandle<int>((int*)intPointer1);
        var handle2 = new TypedMemoryHandle<int>((int*)intPointer2);

        var equalityMethodResult = handle1.Equals(handle2);
        var equalityOperatorResult = handle1 == handle2;
        var inequalityOperatorResult = handle1 != handle2;

        equalityMethodResult.Should().Be(expectedResult);
        equalityOperatorResult.Should().Be(expectedResult);
        inequalityOperatorResult.Should().Be(!expectedResult);
    }

    [Fact]
    public unsafe void Equals_WhenCalledWithObject_ReturnsFalse()
    {
        var handle = new TypedMemoryHandle<int>((int*)0);

        var result = handle.Equals(new object());

        result.Should().BeFalse();
    }

    [Theory]
    [InlineData(0, 0, true)]
    [InlineData(0, 1, false)]
    [InlineData(1, 0, false)]
    [InlineData(1, 1, true)]
    public unsafe void Equals_WhenCalledWithBoxedValue_ReturnsExpectedResult(int left, int right, bool expected)
    {
        var handle = new TypedMemoryHandle<int>((int*)left);

        var result = handle.Equals((object)new TypedMemoryHandle<int>((int*)right));

        result.Should().Be(expected);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public unsafe void GetHashCode_WhenCalled_ReturnsExpectedResult(int intPointer)
    {
        var handle = new TypedMemoryHandle<int>((int*)intPointer);

        var result = handle.GetHashCode();

        result.Should().Be(intPointer);
    }
}