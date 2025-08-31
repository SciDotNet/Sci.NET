// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Differentiation;

public class ContractionFunctionBackwardsTests
{
    [Fact]
    public void Contract_Contracts10_ReturnsCorrectResult()
    {
        SciDotNetConfiguration.PreviewFeatures.EnableAutoGrad();

        // Arrange
        using var tensor1 = Tensor.FromArray<int>(Enumerable.Range(0, 3 * 4 * 5).ToArray(), requiresGradient: true).Reshape(3, 4, 5);
        using var tensor2 = Tensor.FromArray<int>(Enumerable.Range(0, 4 * 5 * 2).ToArray(), requiresGradient: true).Reshape(4, 5, 2);

        // Act
        var result = tensor1.Contract(tensor2, new[] { 1 }, new[] { 0 });
        result.Backward();

        // Assert
        tensor1.Gradient!.Should().NotBeNull();
        tensor2.Gradient!.Should().NotBeNull();

        tensor1
            .Gradient?.Should()
            .HaveEquivalentElements(
                new int[,,]
                {
                    { { 45, 45, 45, 45, 45 }, { 145, 145, 145, 145, 145 }, { 245, 245, 245, 245, 245 }, { 345, 345, 345, 345, 345 } },
                    { { 45, 45, 45, 45, 45 }, { 145, 145, 145, 145, 145 }, { 245, 245, 245, 245, 245 }, { 345, 345, 345, 345, 345 } },
                    { { 45, 45, 45, 45, 45 }, { 145, 145, 145, 145, 145 }, { 245, 245, 245, 245, 245 }, { 345, 345, 345, 345, 345 } }
                });

        tensor2
            .Gradient?.Should()
            .HaveEquivalentElements(
                new int[,,]
                {
                    { { 330, 330 }, { 330, 330 }, { 330, 330 }, { 330, 330 }, { 330, 330 } },
                    { { 405, 405 }, { 405, 405 }, { 405, 405 }, { 405, 405 }, { 405, 405 } },
                    { { 480, 480 }, { 480, 480 }, { 480, 480 }, { 480, 480 }, { 480, 480 } },
                    { { 555, 555 }, { 555, 555 }, { 555, 555 }, { 555, 555 }, { 555, 555 } }
                });
    }
}