// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.VectorOperations;

public class CosineDistanceShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ComputeCosineDistance(IDevice device)
    {
        var left = Tensor.FromArray<float>(new float[] { 0.25f, 0.5f, 0.75f, 1.0f }).ToVector();
        var right = Tensor.FromArray<float>(new float[] { 1.0f, 0.75f, 0.5f, 0.25f }).ToVector();

        left.To(device);
        right.To(device);

        var result = left.CosineDistance(right);

        result.Value.Should().BeApproximately(0.3333333f, 1e-6f);
    }
}