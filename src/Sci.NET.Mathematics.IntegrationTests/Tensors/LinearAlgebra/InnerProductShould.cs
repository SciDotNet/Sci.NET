// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.LinearAlgebra;

public class InnerProductShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ComputeInnerProduct(IDevice device)
    {
        using var left = Tensor.FromArray<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }).WithGradient().ToVector();
        using var right = Tensor.FromArray<float>(new float[] { 8, 7, 6, 5, 4, 3, 2, 1 }).WithGradient().ToVector();

        left.To(device);
        right.To(device);

        using var result = left.Inner(right);

        result.Backward();

        result.Value.Should().Be(120);

        left.Gradient!.Should().NotBeNull();
        left.Gradient!.Should().HaveApproximatelyEquivalentElements(new float[] { 8, 7, 6, 5, 4, 3, 2, 1 }, 1e-6f);

        right.Gradient!.Should().NotBeNull();
        right.Gradient!.Should().HaveApproximatelyEquivalentElements(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, 1e-6f);
    }
}