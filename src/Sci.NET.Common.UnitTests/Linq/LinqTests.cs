// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Extensions;

namespace Sci.NET.Common.UnitTests.Linq;

public class LinqTests
{
    [Theory]
    [InlineData(new int[] { 1, 2, 3, 4, 5 }, 120)]
    [InlineData(new int[] { 1, 2, 3, 4, 5, 6 }, 720)]
    [InlineData(new int[] { 1, 2, 3, 4, 5, 6, 7 }, 5040)]
    [InlineData(new int[] { 1, 2, 3, 4, 5, 6, 7, 8 }, 40320)]
    [InlineData(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, 362880)]
    [InlineData(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, 3628800)]
    public void ProductTest_Int(int[] array, int expected)
    {
        var actual = array.Product();
        actual.Should().Be(expected);
    }

    [Theory]
    [InlineData(new float[] { 1, 2, 3, 4, 5 }, 120)]
    [InlineData(new float[] { 1, 2, 3, 4, 5, 6 }, 720)]
    [InlineData(new float[] { 1, 2, 3, 4, 5, 6, 7 }, 5040)]
    [InlineData(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, 40320)]
    [InlineData(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, 362880)]
    [InlineData(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, 3628800)]
    public void ProductTest_Float(float[] array, float expected)
    {
        var actual = array.Product();
        actual.Should().Be(expected);
    }

    [Theory]
    [InlineData(new double[] { 1, 2, 3, 4, 5 }, 120)]
    [InlineData(new double[] { 1, 2, 3, 4, 5, 6 }, 720)]
    [InlineData(new double[] { 1, 2, 3, 4, 5, 6, 7 }, 5040)]
    [InlineData(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }, 40320)]
    [InlineData(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, 362880)]
    [InlineData(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, 3628800)]
    public void ProductTest_Double(double[] array, double expected)
    {
        var actual = array.Product();
        actual.Should().Be(expected);
    }
}