// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Manipulation.Permutation;

public class PermuteShould
{
    [Fact]
    public void ReturnCorrectValues_GivenValidExample_1()
    {
        var source = Tensor
            .FromArray<int>(Enumerable.Range(0, 24).ToArray())
            .Reshape(4, 3, 2);

        var result = source.Permute(new int[] { 1, 2, 0 });

        result
            .Should()
            .HaveShape(3, 2, 4)
            .And
            .HaveEquivalentElements(new int[,,] { { { 0, 6, 12, 18 }, { 1, 7, 13, 19 } }, { { 2, 8, 14, 20 }, { 3, 9, 15, 21 } }, { { 4, 10, 16, 22 }, { 5, 11, 17, 23 } } });
    }

    [Fact]
    public void ReturnCorrectValues_GivenValidExample_2()
    {
        var source = Tensor
            .FromArray<int>(new int[,,] { { { 10, 11 }, { 12, 13 }, { 14, 15 }, { 16, 17 } }, { { 18, 19 }, { 20, 21 }, { 22, 23 }, { 24, 25 } } });

        var result = source.Permute(new int[] { 2, 1, 0 });

        result
            .Should()
            .HaveShape(2, 4, 2)
            .And
            .HaveEquivalentElements(new int[,,] { { { 10, 18 }, { 12, 20 }, { 14, 22 }, { 16, 24 } }, { { 11, 19 }, { 13, 21 }, { 15, 23 }, { 17, 25 } } });
    }

    [Fact]
    public void ReturnCorrectValues_GivenValidExample_3()
    {
        var source = Tensor
            .FromArray<int>(Enumerable.Range(0, 8).ToArray())
            .Reshape(2, 4);

        var result = source.Permute(new int[] { 1, 0 });

        result
            .Should()
            .HaveShape(4, 2)
            .And
            .HaveEquivalentElements(new int[,] { { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 } });
    }

    [Fact]
    public void ThrowException_GivenIncorrectNumberOfPermutationIndices()
    {
        var source = Tensor
            .FromArray<int>(Enumerable.Range(0, 8).ToArray())
            .Reshape(2, 4);

        var act = () => source.Permute(new int[] { 2, 1, 0 });

        act.Should().Throw<ArgumentException>();
    }
}