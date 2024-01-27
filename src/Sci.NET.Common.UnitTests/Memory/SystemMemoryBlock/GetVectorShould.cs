// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Numerics.Intrinsics;

namespace Sci.NET.Common.UnitTests.Memory.SystemMemoryBlock;

public class GetVectorShould
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void GetCorrectly_GivenInt(int offset)
    {
        var data = new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var block = new SystemMemoryBlock<int>(data.Length);

        block.CopyFrom(data);

        var result = block.UnsafeGetVectorUnchecked<int>(offset);

        for (var i = offset; i < SimdVector.Count<int>(); i++)
        {
            result[i - offset].Should().Be(data[i]);
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void GetCorrectly_GivenFloat(int offset)
    {
        var data = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var block = new SystemMemoryBlock<float>(data.Length);

        block.CopyFrom(data);

        var result = block.UnsafeGetVectorUnchecked<float>(offset);

        for (var i = offset; i < Vector<float>.Count; i++)
        {
            result[i - offset].Should().Be(data[i]);
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void GetCorrectly_GivenDouble(int offset)
    {
        var data = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var block = new SystemMemoryBlock<double>(data.Length);

        block.CopyFrom(data);

        var result = block.UnsafeGetVectorUnchecked<double>(offset);

        for (var i = offset; i < SimdVector.Count<double>(); i++)
        {
            result[i - offset].Should().Be(data[i]);
        }
    }
}