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
        // Arrange
        using var tensor1 = Tensor.FromArray<int>(Enumerable.Range(0, 3 * 4 * 5).ToArray(), requiresGradient: true).Reshape(3, 4, 5);
        using var tensor2 = Tensor.FromArray<int>(Enumerable.Range(0, 4 * 5 * 2).ToArray(), requiresGradient: false).Reshape(4, 5, 2);

        // Left shape: 3,4,5
        // Right shape: 4,5,2
        // Output shape: 3,5,5,2
        // Left Grad Shape: 3,2,2
        // Right Grad Shape: 4,5,2

        // Act
        var result = tensor1.Contract(tensor2, new[] { 1 }, new[] { 0 });
        result.Backward();

        // Assert
        tensor1.Gradient!.Should().NotBeNull();

        tensor1
            .Gradient?.Should()
            .HaveEquivalentElements(
                new int[,,]
                {
                    { { 45, 45, 45, 45, 45 }, { 145, 145, 145, 145, 145 }, { 245, 245, 245, 245, 245 }, { 345, 345, 345, 345, 345 } },
                    { { 45, 45, 45, 45, 45 }, { 145, 145, 145, 145, 145 }, { 245, 245, 245, 245, 245 }, { 345, 345, 345, 345, 345 } },
                    { { 45, 45, 45, 45, 45 }, { 145, 145, 145, 145, 145 }, { 245, 245, 245, 245, 245 }, { 345, 345, 345, 345, 345 } }
                });
    }
}