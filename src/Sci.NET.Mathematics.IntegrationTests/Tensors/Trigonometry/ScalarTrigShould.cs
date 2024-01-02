// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Trigonometry;

public class ScalarTrigShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectResult_ForSin(IDevice device)
    {
        // Sin(0) = 0
        InvokeTest<float>(0.0f, device, x => x.Sin()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.Sin()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.Sin()).Should().BeApproximately(0.0f, 1e-3f);

        // Sin(π/2) = 1
        InvokeTest<float>(float.Pi / 2.0f, device, x => x.Sin()).Should().BeApproximately(1.0f, 1e-6f);
        InvokeTest<double>(double.Pi / 2.0, device, x => x.Sin()).Should().BeApproximately(1.0, 1e-6);
        InvokeTest<BFloat16>(BFloat16.Pi / 2.0f, device, x => x.Sin()).Should().BeApproximately(1.0f, 1e-3f);

        // Sin(π) = 0
        InvokeTest<float>(float.Pi, device, x => x.Sin()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(double.Pi, device, x => x.Sin()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(BFloat16.Pi, device, x => x.Sin()).Should().BeApproximately(0.0f, 1e-3f);

        // Sin(3π/2) = -1
        InvokeTest<float>(3.0f * float.Pi / 2.0f, device, x => x.Sin()).Should().BeApproximately(-1.0f, 1e-6f);
        InvokeTest<double>(3.0 * double.Pi / 2.0, device, x => x.Sin()).Should().BeApproximately(-1.0, 1e-6);
        InvokeTest<BFloat16>(3.0f * BFloat16.Pi / 2.0f, device, x => x.Sin()).Should().BeApproximately(-1.0f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectResult_ForCos(IDevice device)
    {
        // Cos(0) = 1
        InvokeTest<float>(0.0f, device, x => x.Cos()).Should().BeApproximately(1.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.Cos()).Should().BeApproximately(1.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.Cos()).Should().BeApproximately(1.0f, 1e-3f);

        // Cos(π/2) = 0
        InvokeTest<float>(float.Pi / 2.0f, device, x => x.Cos()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(double.Pi / 2.0, device, x => x.Cos()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(BFloat16.Pi / 2.0f, device, x => x.Cos()).Should().BeApproximately(0.0f, 1e-3f);

        // Cos(π) = -1
        InvokeTest<float>(float.Pi, device, x => x.Cos()).Should().BeApproximately(-1.0f, 1e-6f);
        InvokeTest<double>(double.Pi, device, x => x.Cos()).Should().BeApproximately(-1.0, 1e-6);
        InvokeTest<BFloat16>(BFloat16.Pi, device, x => x.Cos()).Should().BeApproximately(-1.0f, 1e-3f);

        // Cos(3π/2) = 0
        InvokeTest<float>(3.0f * float.Pi / 2.0f, device, x => x.Cos()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(3.0 * Math.PI / 2.0, device, x => x.Cos()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(3.0f * BFloat16.Pi / 2.0f, device, x => x.Cos()).Should().BeApproximately(0.0f, 1e-2f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectResult_ForTan(IDevice device)
    {
        // Tan(0) = 0
        InvokeTest<float>(0.0f, device, x => x.Tan()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.Tan()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.Tan()).Should().BeApproximately(0.0f, 1e-3f);

        // Tan(π/4) = 1
        InvokeTest<float>(float.Pi / 4.0f, device, x => x.Tan()).Should().BeApproximately(1.0f, 1e-6f);
        InvokeTest<double>(double.Pi / 4.0, device, x => x.Tan()).Should().BeApproximately(1.0, 1e-6);
        InvokeTest<BFloat16>(BFloat16.Pi / 4.0f, device, x => x.Tan()).Should().BeApproximately(1.0f, 1e-3f);

        // Tan(3π/4) = -1
        InvokeTest<float>(3.0f * float.Pi / 4.0f, device, x => x.Tan()).Should().BeApproximately(-1.0f, 1e-6f);
        InvokeTest<double>(3.0 * double.Pi / 4.0, device, x => x.Tan()).Should().BeApproximately(-1.0, 1e-6);
        InvokeTest<BFloat16>(3.0f * BFloat16.Pi / 4.0f, device, x => x.Tan()).Should().BeApproximately(-1.0f, 1e-2f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForSin2(IDevice device)
    {
        // Sin2(0) = 0
        InvokeTest<float>(0.0f, device, x => x.Sin2()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.Sin2()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.Sin2()).Should().BeApproximately(0.0f, 1e-3f);

        // Sin2(π/2) = 1
        InvokeTest<float>(float.Pi / 2.0f, device, x => x.Sin2()).Should().BeApproximately(1.0f, 1e-6f);
        InvokeTest<double>(double.Pi / 2.0, device, x => x.Sin2()).Should().BeApproximately(1.0, 1e-6);
        InvokeTest<BFloat16>(BFloat16.Pi / 2.0f, device, x => x.Sin2()).Should().BeApproximately(1.0f, 1e-3f);

        // Sin2(π) = 0
        InvokeTest<float>(float.Pi, device, x => x.Sin2()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(double.Pi, device, x => x.Sin2()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(BFloat16.Pi, device, x => x.Sin2()).Should().BeApproximately(0.0f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForCos2(IDevice device)
    {
        // Cos2(0) = 1
        InvokeTest<float>(0.0f, device, x => x.Cos2()).Should().BeApproximately(1.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.Cos2()).Should().BeApproximately(1.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.Cos2()).Should().BeApproximately(1.0f, 1e-3f);

        // Cos2(π/2) = 0
        InvokeTest<float>(float.Pi / 2.0f, device, x => x.Cos2()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(double.Pi / 2.0, device, x => x.Cos2()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(BFloat16.Pi / 2.0f, device, x => x.Cos2()).Should().BeApproximately(0.0f, 1e-3f);

        // Cos2(π) = 1
        InvokeTest<float>(float.Pi, device, x => x.Cos2()).Should().BeApproximately(1.0f, 1e-6f);
        InvokeTest<double>(double.Pi, device, x => x.Cos2()).Should().BeApproximately(1.0, 1e-6);
        InvokeTest<BFloat16>(BFloat16.Pi, device, x => x.Cos2()).Should().BeApproximately(1.0f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForTan2(IDevice device)
    {
        // Tan2(0) = 0
        InvokeTest<float>(0.0f, device, x => x.Tan2()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.Tan2()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.Tan2()).Should().BeApproximately(0.0f, 1e-3f);

        // Tan2(π/4) = 1
        InvokeTest<float>(float.Pi / 4.0f, device, x => x.Tan2()).Should().BeApproximately(1.0f, 1e-6f);
        InvokeTest<double>(double.Pi / 4.0, device, x => x.Tan2()).Should().BeApproximately(1.0, 1e-6);
        InvokeTest<BFloat16>(BFloat16.Pi / 4.0f, device, x => x.Tan2()).Should().BeApproximately(1.0f, 1e-3f);

        // Tan2(3π/4) = 1
        InvokeTest<float>(3.0f * float.Pi / 4.0f, device, x => x.Tan2()).Should().BeApproximately(1.0f, 1e-6f);
        InvokeTest<double>(3.0 * double.Pi / 4.0, device, x => x.Tan2()).Should().BeApproximately(1.0, 1e-6);
        InvokeTest<BFloat16>(3.0f * BFloat16.Pi / 4.0f, device, x => x.Tan2()).Should().BeApproximately(1.0f, 1e-1f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForSinh(IDevice device)
    {
        // Sinh(0) = 0
        InvokeTest<float>(0.0f, device, x => x.Sinh()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.Sinh()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.Sinh()).Should().BeApproximately(0.0f, 1e-3f);

        // Sinh(1) = 1.1752011936438014568823818505956
        InvokeTest<float>(1.0f, device, x => x.Sinh()).Should().BeApproximately(1.1752011936438014568823818505956f, 1e-6f);
        InvokeTest<double>(1.0, device, x => x.Sinh()).Should().BeApproximately(1.1752011936438014568823818505956, 1e-6);
        InvokeTest<BFloat16>(1.0f, device, x => x.Sinh()).Should().BeApproximately(1.1752011936438014568823818505956f, 1e-3f);

        // Sinh(-1) = -1.1752011936438014568823818505956
        InvokeTest<float>(-1.0f, device, x => x.Sinh()).Should().BeApproximately(-1.1752011936438014568823818505956f, 1e-6f);
        InvokeTest<double>(-1.0, device, x => x.Sinh()).Should().BeApproximately(-1.1752011936438014568823818505956, 1e-6);
        InvokeTest<BFloat16>(-1.0f, device, x => x.Sinh()).Should().BeApproximately(-1.1752011936438014568823818505956f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForCosh(IDevice device)
    {
        // Cosh(0) = 1
        InvokeTest<float>(0.0f, device, x => x.Cosh()).Should().BeApproximately(1.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.Cosh()).Should().BeApproximately(1.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.Cosh()).Should().BeApproximately(1.0f, 1e-3f);

        // Cosh(1) = 1.5430806348152437784779056207571
        InvokeTest<float>(1.0f, device, x => x.Cosh()).Should().BeApproximately(1.5430806348152437784779056207571f, 1e-6f);
        InvokeTest<double>(1.0, device, x => x.Cosh()).Should().BeApproximately(1.5430806348152437784779056207571, 1e-6);
        InvokeTest<BFloat16>(1.0f, device, x => x.Cosh()).Should().BeApproximately(1.5430806348152437784779056207571f, 1e-3f);

        // Cosh(-1) = 1.5430806348152437784779056207571
        InvokeTest<float>(-1.0f, device, x => x.Cosh()).Should().BeApproximately(1.5430806348152437784779056207571f, 1e-6f);
        InvokeTest<double>(-1.0, device, x => x.Cosh()).Should().BeApproximately(1.5430806348152437784779056207571, 1e-6);
        InvokeTest<BFloat16>(-1.0f, device, x => x.Cosh()).Should().BeApproximately(1.5430806348152437784779056207571f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForTanh(IDevice device)
    {
        // Tanh(0) = 0
        InvokeTest<float>(0.0f, device, x => x.Tanh()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.Tanh()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.Tanh()).Should().BeApproximately(0.0f, 1e-3f);

        // Tanh(1) = 0.76159415595576488811945828260479
        InvokeTest<float>(1.0f, device, x => x.Tanh()).Should().BeApproximately(0.76159415595576488811945828260479f, 1e-6f);
        InvokeTest<double>(1.0, device, x => x.Tanh()).Should().BeApproximately(0.76159415595576488811945828260479, 1e-6);
        InvokeTest<BFloat16>(1.0f, device, x => x.Tanh()).Should().BeApproximately(0.76159415595576488811945828260479f, 1e-3f);

        // Tanh(-1) = -0.76159415595576488811945828260479
        InvokeTest<float>(-1.0f, device, x => x.Tanh()).Should().BeApproximately(-0.76159415595576488811945828260479f, 1e-6f);
        InvokeTest<double>(-1.0, device, x => x.Tanh()).Should().BeApproximately(-0.76159415595576488811945828260479, 1e-6);
        InvokeTest<BFloat16>(-1.0f, device, x => x.Tanh()).Should().BeApproximately(-0.76159415595576488811945828260479f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForSinh2(IDevice device)
    {
        // Sinh2(0) = 0
        InvokeTest<float>(0.0f, device, x => x.Sinh2()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.Sinh2()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.Sinh2()).Should().BeApproximately(0.0f, 1e-3f);

        // Sinh2(1) = 1.38109784554
        InvokeTest<float>(1.0f, device, x => x.Sinh2()).Should().BeApproximately(1.38109784554f, 1e-4f);
        InvokeTest<double>(1.0, device, x => x.Sinh2()).Should().BeApproximately(1.38109784554, 1e-6);
        InvokeTest<BFloat16>(1.0f, device, x => x.Sinh2()).Should().Be(1.375f);

        // Sinh2(-1) = 1.38109784554
        InvokeTest<float>(-1.0f, device, x => x.Sinh2()).Should().BeApproximately(1.38109784554f, 1e-3f);
        InvokeTest<double>(-1.0, device, x => x.Sinh2()).Should().BeApproximately(1.38109784554, 1e-6);
        InvokeTest<BFloat16>(-1.0f, device, x => x.Sinh2()).Should().Be(1.375f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForCosh2(IDevice device)
    {
        // Cosh2(0) = 1
        InvokeTest<float>(0.0f, device, x => x.Cosh2()).Should().BeApproximately(1.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.Cosh2()).Should().BeApproximately(1.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.Cosh2()).Should().BeApproximately(1.0f, 1e-3f);

        // Cosh2(1) = 2.38109784554
        InvokeTest<float>(1.0f, device, x => x.Cosh2()).Should().BeApproximately(2.38109784554f, 1e-6f);
        InvokeTest<double>(1.0, device, x => x.Cosh2()).Should().BeApproximately(2.38109784554, 1e-6);
        InvokeTest<BFloat16>(1.0f, device, x => x.Cosh2()).Should().Be(2.390625f);

        // Cosh2(-1) = 2.38109784554
        InvokeTest<float>(-1.0f, device, x => x.Cosh2()).Should().BeApproximately(2.38109784554f, 1e-6f);
        InvokeTest<double>(-1.0, device, x => x.Cosh2()).Should().BeApproximately(2.38109784554, 1e-6);
        InvokeTest<BFloat16>(-1.0f, device, x => x.Cosh2()).Should().Be(2.390625f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForTanh2(IDevice device)
    {
        // Tanh2(0) = 0
        InvokeTest<float>(0.0f, device, x => x.Tanh2()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.Tanh2()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.Tanh2()).Should().BeApproximately(0.0f, 1e-3f);

        // Tanh2(1) = 0.58002565838
        InvokeTest<float>(1.0f, device, x => x.Tanh2()).Should().BeApproximately(0.58002565838f, 1e-6f);
        InvokeTest<double>(1.0, device, x => x.Tanh2()).Should().BeApproximately(0.58002565838, 1e-6);
        InvokeTest<BFloat16>(1.0f, device, x => x.Tanh2()).Should().Be(0.58203125f);

        // Tanh2(-1) = 0.58002565838
        InvokeTest<float>(-1.0f, device, x => x.Tanh2()).Should().BeApproximately(0.58002565838f, 1e-6f);
        InvokeTest<double>(-1.0, device, x => x.Tanh2()).Should().BeApproximately(0.58002565838, 1e-6);
        InvokeTest<BFloat16>(-1.0f, device, x => x.Tanh2()).Should().Be(0.58203125f);
    }

    private static TNumber InvokeTest<TNumber>(TNumber value, IDevice device, Func<Scalar<TNumber>, Scalar<TNumber>> function)
        where TNumber : unmanaged, ITrigonometricFunctions<TNumber>, INumber<TNumber>
    {
        var tensor = new Scalar<TNumber>(value);
        tensor.To(device);

        return function(tensor).Value;
    }
}