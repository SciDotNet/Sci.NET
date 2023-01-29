// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Mathematics.Tensors.LinearAlgebra;
using Sci.NET.Test.Common.Assertions.Memory;

namespace Sci.NET.Mathematics.UnitTests.Tensors.Backends.Managed.Ops.LinearAlgebra;

[SuppressMessage(
    "Performance",
    "CA1814:Prefer jagged arrays over multidimensional",
    Justification = "This is a test class.")]
public class ContractionOperationTests
{
    [Fact]
    public void ContractExample1()
    {
        var tensor1 = Tensor.FromArray<float>(
            new float[,,]
            {
                {
                    {
                        1f, 2f, 3f, 4f
                    },
                    {
                        5f, 6f, 7f, 8f
                    },
                    {
                        9f, 10f, 11f, 12f
                    }
                },
                {
                    {
                        13f, 14f, 15f, 16f
                    },
                    {
                        17f, 18f, 19f, 20f
                    },
                    {
                        21f, 22f, 23f, 24f
                    }
                }
            });

        var tensor2 = Tensor.FromArray<float>(
            new float[,]
            {
                {
                    1f, 2f, 3f
                },
                {
                    4f, 5f, 6f
                },
                {
                    7f, 8f, 9f
                },
                {
                    10f, 11f, 12f
                }
            });

        var result = tensor1.TensorDot(
            tensor2,
            new int[]
            {
                2
            },
            new int[]
            {
                0
            });

        result.Data.Should().BeEqualTo(
            new float[]
            {
                70, 80, 90, 158, 184, 210, 246, 288, 330, 334, 392, 450, 422, 496, 570, 510, 600, 690
            });
    }
}