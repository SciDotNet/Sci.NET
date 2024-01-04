// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using JetBrains.dotMemoryUnit;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Xunit.Abstractions;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.LinearAlgebra;

public class ContractShould : IntegrationTestBase
{
    private readonly ITestOutputHelper _output;

    public ContractShould(ITestOutputHelper output)
    {
        _output = output;

        DotMemoryUnitTestOutput.SetOutputMethod(_output.WriteLine);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_GivenTwoTensorsWithSameShape(IDevice device)
    {
        var left = Tensor.FromArray<float>(new float[,,] { { { 0, 1 }, { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 }, { 10, 11 } }, { { 12, 13 }, { 14, 15 }, { 16, 17 } }, { { 18, 19 }, { 20, 21 }, { 22, 23 } } });
        var right = Tensor.FromArray<float>(new float[,,] { { { 0, 1 }, { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 }, { 10, 11 } }, { { 12, 13 }, { 14, 15 }, { 16, 17 } }, { { 18, 19 }, { 20, 21 }, { 22, 23 } } });

        left.To(device);
        right.To(device);

        var result = left.Contract(right, new int[] { 1, 2 }, new int[] { 1, 2 });

        result
            .Should()
            .HaveShape(4, 4)
            .And
            .HaveEquivalentElements(new float[,] { { 55, 145, 235, 325 }, { 145, 451, 757, 1063 }, { 235, 757, 1279, 1801 }, { 325, 1063, 1801, 2539 } });
    }
}