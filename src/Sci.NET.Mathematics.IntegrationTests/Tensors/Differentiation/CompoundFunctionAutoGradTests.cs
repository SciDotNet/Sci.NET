// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Differentiation;

public class CompoundFunctionAutoGradTests
{
    public CompoundFunctionAutoGradTests()
    {
        SciDotNetConfiguration.PreviewFeatures.EnableAutoGrad();
    }

    [Fact]
    public void CosineSinTest()
    {
        // Used PyTorch to verify the results
        // Arrange
        using var tensor = Tensor.FromArray<double>(new double[] { 1, 2, 3, 4 }, requiresGradient: true);

        // Act
        var a = tensor.Cos();
        var b = a.Sin();
        b.Backward();

        // Assert
        tensor.Gradient!.Should().NotBeNull();
        a.Gradient!.Should().NotBeNull();
        b.Gradient!.Should().NotBeNull();

        tensor.Gradient?.Should().HaveApproximatelyEquivalentElements(new double[] { -0.7216061490634432, -0.831691915635113, -0.07743200279648704, 0.6008054073330641 }, 1e-9);
        a.Gradient?.Should().HaveApproximatelyEquivalentElements(new double[] { 0.8575532158463933, 0.9146533258523714, 0.5486961336030971, 0.7938734492261525 }, 1e-9);
        b.Gradient?.Should().HaveApproximatelyEquivalentElements(new double[] { 1, 1, 1, 1 }, 1e-9);
    }

    [Fact]
    public void ReduceAddTests()
    {
        using var tensor = Tensor.FromArray<double>(new double[] { 1, 2, 3, 4 }, requiresGradient: true);

        var a = tensor.Sin().Mean();

        a.Backward();
    }
}