// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Attributes;
using Sci.NET.Accelerators.Runtime;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Accelerators.IntegrationTests.Kernels;

public class VectorAdd : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void VectorAddKernel_ReturnsExpectedResult(IDevice device)
    {
        // Arrange
        var left = Tensor.FromArray<float>(new float[] { 1f, 2f, 3f, 4f, 5f });
        var right = Tensor.FromArray<float>(new float[] { 1f, 2f, 3f, 4f, 5f });
        var expected = Tensor.FromArray<float>(new float[] { 2f, 4f, 6f, 8f, 10f });

        left.To(device);
        right.To(device);
        expected.To(device);

        // Act
        // TODO: Implement a way to run kernels on the device.
    }

    [Kernel]
    private static unsafe void VectorAddKernel(float* left, float* right, float* result, long length)
    {
        if (ParallelThread.ThreadIdx.X < length)
        {
            result[ParallelThread.ThreadIdx.X] = left[ParallelThread.ThreadIdx.X] + right[ParallelThread.ThreadIdx.X];
        }
    }
}