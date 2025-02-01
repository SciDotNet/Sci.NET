// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Manipulation.Transpose;

public class TransposeShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrix(IDevice device)
    {
        var matrix = Tensor.FromArray<int>(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } }).ToMatrix();
        matrix.To(device);

        var transposedMatrix = matrix.Transpose();

        transposedMatrix.To<CpuComputeDevice>();

        transposedMatrix
            .Should()
            .HaveShape(3, 2)
            .And
            .HaveEquivalentElements(new int[,] { { 1, 4 }, { 2, 5 }, { 3, 6 } });
    }
}