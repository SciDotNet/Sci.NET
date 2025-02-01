// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors;

public static class PyTorchTestHelpers
{
    public static void TestForwardAndBackwardPytorchExampleBinaryOp<TNumber>(string inputFile, IDevice device, Func<ITensor<TNumber>, ITensor<TNumber>, ITensor<TNumber>> function)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var loadDirectory = $@"{Path.GetDirectoryName(typeof(PyTorchTestHelpers).Assembly.Location)}\Tensors\";
        var tensors = Tensor.LoadSafeTensors<TNumber>(Path.Combine(loadDirectory, inputFile));
        var left = tensors["left"].ToTensor();
        var right = tensors["right"].ToTensor();
        var expected = tensors["result"].ToTensor();
        var expectedLeftGradient = tensors["left_grad"].ToTensor();
        var expectedRightGradient = tensors["right_grad"].ToTensor();
        using var resultGradient = Tensor.Ones<TNumber>(expected.Shape);

        left.To(device);
        right.To(device);

        var result = function(left, right);
        result.Backward();

        result.To<CpuComputeDevice>();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        result.Should().HaveApproximatelyEquivalentElements(expected.ToArray(), TNumber.CreateChecked(1e-4f));
        result.Gradient?.Should().NotBeNull();
        result.Gradient?.Should().HaveApproximatelyEquivalentElements(resultGradient.ToArray(), TNumber.CreateChecked(1e-4f));
        left.Gradient?.Should().NotBeNull();
        left.Gradient?.Should().HaveApproximatelyEquivalentElements(expectedLeftGradient.ToArray(), TNumber.CreateChecked(1e-4f));
        right.Gradient?.Should().NotBeNull();
        right.Gradient?.Should().HaveApproximatelyEquivalentElements(expectedRightGradient.ToArray(), TNumber.CreateChecked(1e-4f));
    }
}