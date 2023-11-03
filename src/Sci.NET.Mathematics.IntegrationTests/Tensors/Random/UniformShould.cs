// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.CUDA.Tensors;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Random;

public class UniformShould : IntegrationTestBase
{
    [Fact]
    public void BeConsistentWithSeed()
    {
        var tensor1 = Tensor.Random.Uniform<float, CudaComputeDevice>(
            new Shape(200, 300, 400),
            2,
            3,
            4);

        tensor1.To<CudaComputeDevice>();

        var tensor2 = Tensor.Random.Uniform<float, CudaComputeDevice>(
            new Shape(200, 300, 400),
            2,
            3,
            4);

        tensor1
            .Should()
            .HaveShape(200, 300, 400)
            .And
            .HaveEquivalentElements(tensor2.ToArray());
    }
}