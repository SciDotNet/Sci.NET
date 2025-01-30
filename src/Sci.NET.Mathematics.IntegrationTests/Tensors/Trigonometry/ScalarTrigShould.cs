// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

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

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForAsin(IDevice device)
    {
        // Asin(0) = 0
        InvokeTest<float>(0.0f, device, x => x.ASin()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.ASin()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.ASin()).Should().BeApproximately(0.0f, 1e-3f);

        // Asin(1) = π/2
        InvokeTest<float>(1.0f, device, x => x.ASin()).Should().BeApproximately(float.Pi / 2.0f, 1e-6f);
        InvokeTest<double>(1.0, device, x => x.ASin()).Should().BeApproximately(double.Pi / 2.0, 1e-6);
        InvokeTest<BFloat16>(1.0f, device, x => x.ASin()).Should().BeApproximately(BFloat16.Pi / 2.0f, 1e-3f);

        // Asin(-1) = -π/2
        InvokeTest<float>(-1.0f, device, x => x.ASin()).Should().BeApproximately(-float.Pi / 2.0f, 1e-6f);
        InvokeTest<double>(-1.0, device, x => x.ASin()).Should().BeApproximately(-double.Pi / 2.0, 1e-6);
        InvokeTest<BFloat16>(-1.0f, device, x => x.ASin()).Should().BeApproximately(-BFloat16.Pi / 2.0f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForAcos(IDevice device)
    {
        // Acos(0) = π/2
        InvokeTest<float>(0.0f, device, x => x.ACos()).Should().BeApproximately(float.Pi / 2.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.ACos()).Should().BeApproximately(double.Pi / 2.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.ACos()).Should().BeApproximately(BFloat16.Pi / 2.0f, 1e-3f);

        // Acos(1) = 0
        InvokeTest<float>(1.0f, device, x => x.ACos()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(1.0, device, x => x.ACos()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(1.0f, device, x => x.ACos()).Should().BeApproximately(0.0f, 1e-3f);

        // Acos(-1) = π
        InvokeTest<float>(-1.0f, device, x => x.ACos()).Should().BeApproximately(float.Pi, 1e-6f);
        InvokeTest<double>(-1.0, device, x => x.ACos()).Should().BeApproximately(double.Pi, 1e-6);
        InvokeTest<BFloat16>(-1.0f, device, x => x.ACos()).Should().BeApproximately(BFloat16.Pi, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForAtan(IDevice device)
    {
        // Atan(0) = 0
        InvokeTest<float>(0.0f, device, x => x.ATan()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.ATan()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.ATan()).Should().BeApproximately(0.0f, 1e-3f);

        // Atan(1) = π/4
        InvokeTest<float>(1.0f, device, x => x.ATan()).Should().BeApproximately(float.Pi / 4.0f, 1e-6f);
        InvokeTest<double>(1.0, device, x => x.ATan()).Should().BeApproximately(double.Pi / 4.0, 1e-6);
        InvokeTest<BFloat16>(1.0f, device, x => x.ATan()).Should().BeApproximately(BFloat16.Pi / 4.0f, 1e-3f);

        // Atan(-1) = -π/4
        InvokeTest<float>(-1.0f, device, x => x.ATan()).Should().BeApproximately(-float.Pi / 4.0f, 1e-6f);
        InvokeTest<double>(-1.0, device, x => x.ATan()).Should().BeApproximately(-double.Pi / 4.0, 1e-6);
        InvokeTest<BFloat16>(-1.0f, device, x => x.ATan()).Should().BeApproximately(-BFloat16.Pi / 4.0f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForASinh(IDevice device)
    {
        // Asinh(0) = 0
        InvokeTest<float>(0.0f, device, x => x.ASinh()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.ASinh()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.ASinh()).Should().BeApproximately(0.0f, 1e-3f);

        // Asinh(1) = 0.88137358701954302523260932497979
        InvokeTest<float>(1.0f, device, x => x.ASinh()).Should().BeApproximately(0.88137358701954302523260932497979f, 1e-6f);
        InvokeTest<double>(1.0, device, x => x.ASinh()).Should().BeApproximately(0.88137358701954302523260932497979, 1e-6);
        InvokeTest<BFloat16>(1.0f, device, x => x.ASinh()).Should().BeApproximately(0.88137358701954302523260932497979f, 1e-3f);

        // Asinh(-1) = -0.88137358701954302523260932497979
        InvokeTest<float>(-1.0f, device, x => x.ASinh()).Should().BeApproximately(-0.88137358701954302523260932497979f, 1e-6f);
        InvokeTest<double>(-1.0, device, x => x.ASinh()).Should().BeApproximately(-0.88137358701954302523260932497979, 1e-6);
        InvokeTest<BFloat16>(-1.0f, device, x => x.ASinh()).Should().BeApproximately(-0.88137358701954302523260932497979f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForACosh(IDevice device)
    {
        // Acosh(1) = 0
        InvokeTest<float>(1.0f, device, x => x.ACosh()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(1.0, device, x => x.ACosh()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(1.0f, device, x => x.ACosh()).Should().BeApproximately(0.0f, 1e-3f);

        // Acosh(2) = 1.31695789692481670862504634730797
        InvokeTest<float>(2.0f, device, x => x.ACosh()).Should().BeApproximately(1.31695789692481670862504634730797f, 1e-6f);
        InvokeTest<double>(2.0, device, x => x.ACosh()).Should().BeApproximately(1.31695789692481670862504634730797, 1e-6);
        InvokeTest<BFloat16>(2.0f, device, x => x.ACosh()).Should().BeApproximately(1.31695789692481670862504634730797f, 1e-3f);

        // Acosh(-1) = NaN
        InvokeTest<float>(-1.0f, device, x => x.ACosh()).Should().Be(float.NaN);
        InvokeTest<double>(-1.0, device, x => x.ACosh()).Should().Be(double.NaN);
        InvokeTest<BFloat16>(-1.0f, device, x => x.ACosh()).Should().Be(BFloat16.NaN);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForATanh(IDevice device)
    {
        // Atanh(0) = 0
        InvokeTest<float>(0.0f, device, x => x.ATanh()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.ATanh()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.ATanh()).Should().BeApproximately(0.0f, 1e-3f);

        // Atanh(0.5) = 0.54930614433405484569762261846126
        InvokeTest<float>(0.5f, device, x => x.ATanh()).Should().BeApproximately(0.54930614433405484569762261846126f, 1e-6f);
        InvokeTest<double>(0.5, device, x => x.ATanh()).Should().BeApproximately(0.54930614433405484569762261846126, 1e-6);
        InvokeTest<BFloat16>(0.5f, device, x => x.ATanh()).Should().BeApproximately(0.54930614433405484569762261846126f, 1e-3f);

        // Atanh(-0.5) = -0.54930614433405484569762261846126
        InvokeTest<float>(-0.5f, device, x => x.ATanh()).Should().BeApproximately(-0.54930614433405484569762261846126f, 1e-6f);
        InvokeTest<double>(-0.5, device, x => x.ATanh()).Should().BeApproximately(-0.54930614433405484569762261846126, 1e-6);
        InvokeTest<BFloat16>(-0.5f, device, x => x.ATanh()).Should().BeApproximately(-0.54930614433405484569762261846126f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForASinh2(IDevice device)
    {
        // Asinh2(0) = 0
        InvokeTest<float>(0.0f, device, x => x.ASinh2()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.ASinh2()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.ASinh2()).Should().BeApproximately(0.0f, 1e-3f);

        // Asinh2(1) = 0.7768193999
        InvokeTest<float>(1.0f, device, x => x.ASinh2()).Should().BeApproximately(0.7768193999f, 1e-6f);
        InvokeTest<double>(1.0, device, x => x.ASinh2()).Should().BeApproximately(0.7768193999, 1e-6);
        InvokeTest<BFloat16>(1.0f, device, x => x.ASinh2()).Should().BeApproximately(0.78125f, 1e-5f);

        // Asinh2(-1) = 0.7768193999
        InvokeTest<float>(-1.0f, device, x => x.ASinh2()).Should().BeApproximately(0.7768193999f, 1e-6f);
        InvokeTest<double>(-1.0, device, x => x.ASinh2()).Should().BeApproximately(0.7768193999, 1e-6);
        InvokeTest<BFloat16>(-1.0f, device, x => x.ASinh2()).Should().BeApproximately(0.78125f, 1e-5f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForACosh2(IDevice device)
    {
        // Acosh2(1) = 0
        InvokeTest<float>(1.0f, device, x => x.ACosh2()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(1.0, device, x => x.ACosh2()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(1.0f, device, x => x.ACosh2()).Should().BeApproximately(0.0f, 1e-3f);

        // Acosh2(2) = 1.734378102
        InvokeTest<float>(2.0f, device, x => x.ACosh2()).Should().BeApproximately(1.734378102f, 1e-6f);
        InvokeTest<double>(2.0, device, x => x.ACosh2()).Should().BeApproximately(1.734378102, 1e-6);
        InvokeTest<BFloat16>(2.0f, device, x => x.ACosh2()).Should().BeApproximately(1.7421875f, 1e-3f);

        // Acosh2(-1) = NaN
        InvokeTest<float>(-1.0f, device, x => x.ACosh2()).Should().Be(float.NaN);
        InvokeTest<double>(-1.0, device, x => x.ACosh2()).Should().Be(double.NaN);
        InvokeTest<BFloat16>(-1.0f, device, x => x.ACosh2()).Should().Be(BFloat16.NaN);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForATanh2(IDevice device)
    {
        // Atanh2(0) = 0
        InvokeTest<float>(0.0f, device, x => x.ATanh2()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.ATanh2()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.ATanh2()).Should().BeApproximately(0.0f, 1e-3f);

        // Atanh2(0.5) = 0.3017372402
        InvokeTest<float>(0.5f, device, x => x.ATanh2()).Should().BeApproximately(0.3017372402f, 1e-6f);
        InvokeTest<double>(0.5, device, x => x.ATanh2()).Should().BeApproximately(0.3017372402, 1e-6);
        InvokeTest<BFloat16>(0.5f, device, x => x.ATanh2()).Should().BeApproximately(0.30273438f, 1e-3f);

        // Atanh2(-0.5) = 0.3017372402
        InvokeTest<float>(-0.5f, device, x => x.ATanh2()).Should().BeApproximately(0.3017372402f, 1e-6f);
        InvokeTest<double>(-0.5, device, x => x.ATanh2()).Should().BeApproximately(0.3017372402, 1e-6);
        InvokeTest<BFloat16>(-0.5f, device, x => x.ATanh2()).Should().BeApproximately(0.30273438f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForASin2(IDevice device)
    {
        // Asin2(0) = 0
        InvokeTest<float>(0.0f, device, x => x.ASin2()).Should().BeApproximately(0.0f, 1e-6f);
        InvokeTest<double>(0.0, device, x => x.ASin2()).Should().BeApproximately(0.0, 1e-6);
        InvokeTest<BFloat16>(0.0f, device, x => x.ASin2()).Should().BeApproximately(0.0f, 1e-3f);

        // Asin2(1) = π/2
        InvokeTest<float>(1.0f, device, x => x.ASin2()).Should().BeApproximately(2.4674013F, 1e-6f);
        InvokeTest<double>(1.0, device, x => x.ASin2()).Should().BeApproximately(2.4674013F, 1e-6);
        InvokeTest<BFloat16>(1.0f, device, x => x.ASin2()).Should().BeApproximately(2.4674013F, 1e-3f);

        // Asin2(-1) = -π/2
        InvokeTest<float>(-1.0f, device, x => x.ASin2()).Should().BeApproximately(2.4674013F, 1e-6f);
        InvokeTest<double>(-1.0, device, x => x.ASin2()).Should().BeApproximately(2.4674013F, 1e-6);
        InvokeTest<BFloat16>(-1.0f, device, x => x.ASin2()).Should().BeApproximately(2.4674013F, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForACos2(IDevice device)
    {
        // Acos2(0) = π/2
        InvokeTest<float>(-1, device, x => x.ACos2()).Should().BeApproximately(9.869605F, 1e-6f);
        InvokeTest<double>(-1, device, x => x.ACos2()).Should().BeApproximately(9.869605F, 1e-6);
        InvokeTest<BFloat16>(-1, device, x => x.ACos2()).Should().BeApproximately(9.869605F, 1e-3f);

        // Acos2(1) = 0
        InvokeTest<float>(0, device, x => x.ACos2()).Should().BeApproximately(2.4674013F, 1e-6f);
        InvokeTest<double>(0, device, x => x.ACos2()).Should().BeApproximately(2.4674013F, 1e-6);
        InvokeTest<BFloat16>(0, device, x => x.ACos2()).Should().BeApproximately(2.4674013F, 1e-3f);

        // Acos2(-1) = π
        InvokeTest<float>(1, device, x => x.ACos2()).Should().BeApproximately(0, 1e-6f);
        InvokeTest<double>(1, device, x => x.ACos2()).Should().BeApproximately(0, 1e-6);
        InvokeTest<BFloat16>(1, device, x => x.ACos2()).Should().BeApproximately(0, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForATan2(IDevice device)
    {
        // Atan2(0) = 0
        InvokeTest<float>(-10, device, x => x.ATan2()).Should().BeApproximately(2.1642165F, 1e-6f);
        InvokeTest<double>(-10, device, x => x.ATan2()).Should().BeApproximately(2.1642166341023152, 1e-6);
        InvokeTest<BFloat16>(-10, device, x => x.ATan2()).Should().BeApproximately(2.15625f, 1e-3f);

        // Atan2(1) = π/4
        InvokeTest<float>(0, device, x => x.ATan2()).Should().BeApproximately(0, 1e-6f);
        InvokeTest<double>(0, device, x => x.ATan2()).Should().BeApproximately(0, 1e-6);
        InvokeTest<BFloat16>(0, device, x => x.ATan2()).Should().BeApproximately(0, 1e-3f);

        // Atan2(-1) = -π/4
        InvokeTest<float>(10, device, x => x.ATan2()).Should().BeApproximately(2.1642165F, 1e-6f);
        InvokeTest<double>(10, device, x => x.ATan2()).Should().BeApproximately(2.1642166341023152, 1e-6);
        InvokeTest<BFloat16>(10, device, x => x.ATan2()).Should().BeApproximately(2.15625f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForSec(IDevice device)
    {
        // Sec(0) = 1
        InvokeTest<float>(0, device, x => x.Sec()).Should().BeApproximately(1, 1e-6f);
        InvokeTest<double>(0, device, x => x.Sec()).Should().BeApproximately(1, 1e-6);
        InvokeTest<BFloat16>(0, device, x => x.Sec()).Should().BeApproximately(1, 1e-3f);

        // Sec(1) = 1.85081571768
        InvokeTest<float>(1, device, x => x.Sec()).Should().BeApproximately(1.85081571768f, 1e-6f);
        InvokeTest<double>(1, device, x => x.Sec()).Should().BeApproximately(1.85081571768, 1e-6);
        InvokeTest<BFloat16>(1, device, x => x.Sec()).Should().BeApproximately(1.8515625f, 1e-3f);

        // Sec(-1) = 1.85081571768
        InvokeTest<float>(-1, device, x => x.Sec()).Should().BeApproximately(1.85081571768f, 1e-6f);
        InvokeTest<double>(-1, device, x => x.Sec()).Should().BeApproximately(1.85081571768, 1e-6);
        InvokeTest<BFloat16>(-1, device, x => x.Sec()).Should().BeApproximately(1.8515625f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForCsc(IDevice device)
    {
        // Csc(-3π/2) = 1
        InvokeTest<float>(-3 * float.Pi / 2, device, x => x.Csc()).Should().BeApproximately(1, 1e-3f);
        InvokeTest<double>(-3 * double.Pi / 2, device, x => x.Csc()).Should().BeApproximately(1, 1e-6);
        InvokeTest<BFloat16>(-3 * BFloat16.Pi / 2, device, x => x.Csc()).Should().BeApproximately(1, 1e-3f);

        // Csc(-π/2) = -1
        InvokeTest<float>(-float.Pi / 2, device, x => x.Csc()).Should().BeApproximately(-1, 1e-3f);
        InvokeTest<double>(-double.Pi / 2, device, x => x.Csc()).Should().BeApproximately(-1, 1e-6);
        InvokeTest<BFloat16>(-BFloat16.Pi / 2, device, x => x.Csc()).Should().BeApproximately(-1, 1e-3f);

        // Csc(0) = ∞
        InvokeTest<float>(0, device, x => x.Csc()).Should().Be(float.PositiveInfinity);
        InvokeTest<double>(0, device, x => x.Csc()).Should().Be(double.PositiveInfinity);
        InvokeTest<BFloat16>(0, device, x => x.Csc()).Should().Be(BFloat16.PositiveInfinity);

        // Csc(π/2) = 1
        InvokeTest<float>(float.Pi / 2, device, x => x.Csc()).Should().BeApproximately(1, 1e-3f);
        InvokeTest<double>(double.Pi / 2, device, x => x.Csc()).Should().BeApproximately(1, 1e-6);
        InvokeTest<BFloat16>(BFloat16.Pi / 2, device, x => x.Csc()).Should().BeApproximately(1, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForCot(IDevice device)
    {
        // Cot(0) = ∞
        InvokeTest<float>(0, device, x => x.Cot()).Should().Be(float.PositiveInfinity);
        InvokeTest<double>(0, device, x => x.Cot()).Should().Be(double.PositiveInfinity);
        InvokeTest<BFloat16>(0, device, x => x.Cot()).Should().Be(BFloat16.PositiveInfinity);

        // Cot(π/4) = 1
        InvokeTest<float>(float.Pi / 4, device, x => x.Cot()).Should().BeApproximately(1, 1e-3f);
        InvokeTest<double>(double.Pi / 4, device, x => x.Cot()).Should().BeApproximately(1, 1e-6);
        InvokeTest<BFloat16>(BFloat16.Pi / 4, device, x => x.Cot()).Should().BeApproximately(1, 1e-3f);

        // Cot(-π/4) = -1
        InvokeTest<float>(-float.Pi / 4, device, x => x.Cot()).Should().BeApproximately(-1, 1e-3f);
        InvokeTest<double>(-double.Pi / 4, device, x => x.Cot()).Should().BeApproximately(-1, 1e-6);
        InvokeTest<BFloat16>(-BFloat16.Pi / 4, device, x => x.Cot()).Should().BeApproximately(-1, 1e-3f);

        // Cot(π/2) = 0
        InvokeTest<float>(float.Pi / 2, device, x => x.Cot()).Should().BeApproximately(0, 1e-3f);
        InvokeTest<double>(double.Pi / 2, device, x => x.Cot()).Should().BeApproximately(0, 1e-6);
        InvokeTest<BFloat16>(BFloat16.Pi / 2, device, x => x.Cot()).Should().BeApproximately(0, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForSec2(IDevice device)
    {
        // Sec2(-pi) = 1
        InvokeTest<float>(-float.Pi, device, x => x.Sec2()).Should().BeApproximately(1, 1e-6f);
        InvokeTest<double>(-double.Pi, device, x => x.Sec2()).Should().BeApproximately(1, 1e-6);
        InvokeTest<BFloat16>(-BFloat16.Pi, device, x => x.Sec2()).Should().BeApproximately(1, 1e-3f);

        // Sec2(0) = 1
        InvokeTest<float>(0, device, x => x.Sec2()).Should().BeApproximately(1, 1e-6f);
        InvokeTest<double>(0, device, x => x.Sec2()).Should().BeApproximately(1, 1e-6);
        InvokeTest<BFloat16>(0, device, x => x.Sec2()).Should().BeApproximately(1, 1e-3f);

        // Sec2(1) = 3.42658477473
        InvokeTest<float>(1, device, x => x.Sec2()).Should().BeApproximately(3.425519F, 1e-6f);
        InvokeTest<double>(1, device, x => x.Sec2()).Should().BeApproximately(3.425518820814759, 1e-6);
        InvokeTest<BFloat16>(1, device, x => x.Sec2()).Should().BeApproximately(3.4375f, 1e-3f);

        // Sec2(-1) = 3.42658477473
        InvokeTest<float>(-1, device, x => x.Sec2()).Should().BeApproximately(3.425519F, 1e-6f);
        InvokeTest<double>(-1, device, x => x.Sec2()).Should().BeApproximately(3.425518820814759, 1e-6);
        InvokeTest<BFloat16>(-1, device, x => x.Sec2()).Should().BeApproximately(3.4375f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForCsc2(IDevice device)
    {
        // Csc2(-3π/2) = 1
        InvokeTest<float>(-3 * float.Pi / 2, device, x => x.Csc2()).Should().BeApproximately(1, 1e-3f);
        InvokeTest<double>(-3 * double.Pi / 2, device, x => x.Csc2()).Should().BeApproximately(1, 1e-6);
        InvokeTest<BFloat16>(-3 * BFloat16.Pi / 2, device, x => x.Csc2()).Should().BeApproximately(1, 1e-3f);

        // Csc2(-π/2) = 1
        InvokeTest<float>(-float.Pi / 2, device, x => x.Csc2()).Should().BeApproximately(1, 1e-3f);
        InvokeTest<double>(-double.Pi / 2, device, x => x.Csc2()).Should().BeApproximately(1, 1e-6);
        InvokeTest<BFloat16>(-BFloat16.Pi / 2, device, x => x.Csc2()).Should().BeApproximately(1, 1e-3f);

        // Csc2(0) = ∞
        InvokeTest<float>(0, device, x => x.Csc2()).Should().Be(float.PositiveInfinity);
        InvokeTest<double>(0, device, x => x.Csc2()).Should().Be(double.PositiveInfinity);
        InvokeTest<BFloat16>(0, device, x => x.Csc2()).Should().Be(BFloat16.PositiveInfinity);

        // Csc2(π/2) = 1
        InvokeTest<float>(float.Pi / 2, device, x => x.Csc2()).Should().BeApproximately(1, 1e-3f);
        InvokeTest<double>(double.Pi / 2, device, x => x.Csc2()).Should().BeApproximately(1, 1e-6);
        InvokeTest<BFloat16>(BFloat16.Pi / 2, device, x => x.Csc2()).Should().BeApproximately(1, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForCot2(IDevice device)
    {
        // Cot2(0) = ∞
        InvokeTest<float>(0, device, x => x.Cot2()).Should().Be(float.PositiveInfinity);
        InvokeTest<double>(0, device, x => x.Cot2()).Should().Be(double.PositiveInfinity);
        InvokeTest<BFloat16>(0, device, x => x.Cot2()).Should().Be(BFloat16.PositiveInfinity);

        // Cot2(π/4) = 1
        InvokeTest<float>(float.Pi / 4, device, x => x.Cot2()).Should().BeApproximately(1, 1e-3f);
        InvokeTest<double>(double.Pi / 4, device, x => x.Cot2()).Should().BeApproximately(1, 1e-6);
        InvokeTest<BFloat16>(BFloat16.Pi / 4, device, x => x.Cot2()).Should().BeApproximately(1, 1e-3f);

        // Cot2(-π/4) = -1
        InvokeTest<float>(-float.Pi / 4, device, x => x.Cot2()).Should().BeApproximately(1, 1e-3f);
        InvokeTest<double>(-double.Pi / 4, device, x => x.Cot2()).Should().BeApproximately(1, 1e-6);
        InvokeTest<BFloat16>(-BFloat16.Pi / 4, device, x => x.Cot2()).Should().BeApproximately(1, 1e-3f);

        // Cot2(π/2) = 0
        InvokeTest<float>(float.Pi / 2, device, x => x.Cot2()).Should().BeApproximately(0, 1e-3f);
        InvokeTest<double>(double.Pi / 2, device, x => x.Cot2()).Should().BeApproximately(0, 1e-6);
        InvokeTest<BFloat16>(BFloat16.Pi / 2, device, x => x.Cot2()).Should().BeApproximately(0, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForSech(IDevice device)
    {
        // Sech(0) = 1
        InvokeTest<float>(0, device, x => x.Sech()).Should().BeApproximately(1, 1e-6f);
        InvokeTest<double>(0, device, x => x.Sech()).Should().BeApproximately(1, 1e-6);
        InvokeTest<BFloat16>(0, device, x => x.Sech()).Should().BeApproximately(1, 1e-3f);

        // Sech(1) = 0.6480542736
        InvokeTest<float>(1, device, x => x.Sech()).Should().BeApproximately(0.6480542736f, 1e-6f);
        InvokeTest<double>(1, device, x => x.Sech()).Should().BeApproximately(0.6480542736, 1e-6);
        InvokeTest<BFloat16>(1, device, x => x.Sech()).Should().BeApproximately(0.64453125f, 1e-3f);

        // Sech(-1) = 0.6480542736
        InvokeTest<float>(-1, device, x => x.Sech()).Should().BeApproximately(0.6480542736f, 1e-6f);
        InvokeTest<double>(-1, device, x => x.Sech()).Should().BeApproximately(0.6480542736, 1e-6);
        InvokeTest<BFloat16>(-1, device, x => x.Sech()).Should().BeApproximately(0.64453125f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForCsch(IDevice device)
    {
        // Csch(0) = ∞
        InvokeTest<float>(0, device, x => x.Csch()).Should().Be(float.PositiveInfinity);
        InvokeTest<double>(0, device, x => x.Csch()).Should().Be(double.PositiveInfinity);
        InvokeTest<BFloat16>(0, device, x => x.Csch()).Should().Be(BFloat16.PositiveInfinity);

        // Csch(1) = 0.850918128
        InvokeTest<float>(1, device, x => x.Csch()).Should().BeApproximately(0.850918128f, 1e-6f);
        InvokeTest<double>(1, device, x => x.Csch()).Should().BeApproximately(0.850918128, 1e-6);
        InvokeTest<BFloat16>(1, device, x => x.Csch()).Should().BeApproximately(0.8515625f, 1e-3f);

        // Csch(-1) = -0.850918128
        InvokeTest<float>(-1, device, x => x.Csch()).Should().BeApproximately(-0.850918128f, 1e-6f);
        InvokeTest<double>(-1, device, x => x.Csch()).Should().BeApproximately(-0.850918128, 1e-6);
        InvokeTest<BFloat16>(-1, device, x => x.Csch()).Should().BeApproximately(-0.8515625f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForCoth(IDevice device)
    {
        // Coth(0) = ∞
        InvokeTest<float>(0, device, x => x.Coth()).Should().Be(float.PositiveInfinity);
        InvokeTest<double>(0, device, x => x.Coth()).Should().Be(double.PositiveInfinity);
        InvokeTest<BFloat16>(0, device, x => x.Coth()).Should().Be(BFloat16.PositiveInfinity);

        // Coth(1) = 1.3130352855
        InvokeTest<float>(1, device, x => x.Coth()).Should().BeApproximately(1.3130352855f, 1e-6f);
        InvokeTest<double>(1, device, x => x.Coth()).Should().BeApproximately(1.3130352855, 1e-6);
        InvokeTest<BFloat16>(1, device, x => x.Coth()).Should().BeApproximately(1.3125f, 1e-2f);

        // Coth(-1) = -1.3130352855
        InvokeTest<float>(-1, device, x => x.Coth()).Should().BeApproximately(-1.3130352855f, 1e-6f);
        InvokeTest<double>(-1, device, x => x.Coth()).Should().BeApproximately(-1.3130352855, 1e-6);
        InvokeTest<BFloat16>(-1, device, x => x.Coth()).Should().BeApproximately(-1.3125f, 1e-2f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForASec(IDevice device)
    {
        // Asec(-1) = π
        InvokeTest<float>(-1, device, x => x.ASec()).Should().BeApproximately(float.Pi, 1e-6f);
        InvokeTest<double>(-1, device, x => x.ASec()).Should().BeApproximately(double.Pi, 1e-6);
        InvokeTest<BFloat16>(-1, device, x => x.ASec()).Should().BeApproximately(BFloat16.Pi, 1e-3f);

        // Asec(1) = 0
        InvokeTest<float>(1, device, x => x.ASec()).Should().BeApproximately(0, 1e-6f);
        InvokeTest<double>(1, device, x => x.ASec()).Should().BeApproximately(0, 1e-6);
        InvokeTest<BFloat16>(1, device, x => x.ASec()).Should().BeApproximately(0, 1e-3f);

        // Asec(1) = 0
        InvokeTest<float>(1, device, x => x.ASec()).Should().BeApproximately(0, 1e-6f);
        InvokeTest<double>(1, device, x => x.ASec()).Should().BeApproximately(0, 1e-6);
        InvokeTest<BFloat16>(1, device, x => x.ASec()).Should().BeApproximately(0, 1e-3f);

        // Asec(2) = 1.047197551
        InvokeTest<float>(2, device, x => x.ASec()).Should().BeApproximately(1.047197551f, 1e-6f);
        InvokeTest<double>(2, device, x => x.ASec()).Should().BeApproximately(1.047197551, 1e-6);
        InvokeTest<BFloat16>(2, device, x => x.ASec()).Should().BeApproximately(1.046875f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForACsc(IDevice device)
    {
        // Acsc(-1) = -π/2
        InvokeTest<float>(-1, device, x => x.ACsc()).Should().BeApproximately(-float.Pi / 2, 1e-6f);
        InvokeTest<double>(-1, device, x => x.ACsc()).Should().BeApproximately(-double.Pi / 2, 1e-6);
        InvokeTest<BFloat16>(-1, device, x => x.ACsc()).Should().BeApproximately(-BFloat16.Pi / 2, 1e-3f);

        // Acsc(1) = π/2
        InvokeTest<float>(1, device, x => x.ACsc()).Should().BeApproximately(float.Pi / 2, 1e-6f);
        InvokeTest<double>(1, device, x => x.ACsc()).Should().BeApproximately(double.Pi / 2, 1e-6);
        InvokeTest<BFloat16>(1, device, x => x.ACsc()).Should().BeApproximately(BFloat16.Pi / 2, 1e-3f);

        // Acsc(2) = 0.5235987756
        InvokeTest<float>(2, device, x => x.ACsc()).Should().BeApproximately(0.5235987756f, 1e-6f);
        InvokeTest<double>(2, device, x => x.ACsc()).Should().BeApproximately(0.5235987756, 1e-6);
        InvokeTest<BFloat16>(2, device, x => x.ACsc()).Should().BeApproximately(0.5234375f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForACot(IDevice device)
    {
        // Acot(1) = π/4
        InvokeTest<float>(1, device, x => x.ACot()).Should().BeApproximately(float.Pi / 4, 1e-6f);
        InvokeTest<double>(1, device, x => x.ACot()).Should().BeApproximately(double.Pi / 4, 1e-6);
        InvokeTest<BFloat16>(1, device, x => x.ACot()).Should().BeApproximately(BFloat16.Pi / 4, 1e-3f);

        // Acot(0) = π/2
        InvokeTest<float>(0, device, x => x.ACot()).Should().BeApproximately(float.Pi / 2, 1e-6f);
        InvokeTest<double>(0, device, x => x.ACot()).Should().BeApproximately(double.Pi / 2, 1e-6);
        InvokeTest<BFloat16>(0, device, x => x.ACot()).Should().BeApproximately(BFloat16.Pi / 2, 1e-3f);

        // Acot(2) = 0.463647609
        InvokeTest<float>(2, device, x => x.ACot()).Should().BeApproximately(0.463647609f, 1e-6f);
        InvokeTest<double>(2, device, x => x.ACot()).Should().BeApproximately(0.463647609, 1e-6);
        InvokeTest<BFloat16>(2, device, x => x.ACot()).Should().BeApproximately(0.46289062f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForAsech(IDevice device)
    {
        // Asech(0.5) = 1.31695789692481670862504634730797
        InvokeTest<float>(0.5f, device, x => x.ASech()).Should().BeApproximately(1.31695789692481670862504634730797f, 1e-6f);
        InvokeTest<double>(0.5, device, x => x.ASech()).Should().BeApproximately(1.31695789692481670862504634730797, 1e-6);
        InvokeTest<BFloat16>(0.5f, device, x => x.ASech()).Should().BeApproximately(1.31695789692481670862504634730797f, 1e-3f);

        // Asech(0.8) = 0.732668414
        InvokeTest<float>(0.8f, device, x => x.ASech()).Should().BeApproximately(0.6931472F, 1e-6f);
        InvokeTest<double>(0.8, device, x => x.ASech()).Should().BeApproximately(0.6931472F, 1e-6);
        InvokeTest<BFloat16>(0.8f, device, x => x.ASech()).Should().BeApproximately(0.6931472F, 1e-3f);

        // Asech(0.2) = 2.292431669
        InvokeTest<float>(0.2f, device, x => x.ASech()).Should().BeApproximately(2.292431669f, 1e-6f);
        InvokeTest<double>(0.2, device, x => x.ASech()).Should().BeApproximately(2.292431669, 1e-6);
        InvokeTest<BFloat16>(0.2f, device, x => x.ASech()).Should().BeApproximately(2.29296875f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForAcsch(IDevice device)
    {
        // Acsch(1) = 0.881373587
        InvokeTest<float>(1.0f, device, x => x.ACsch()).Should().BeApproximately(0.881373587f, 1e-6f);
        InvokeTest<double>(1.0, device, x => x.ACsch()).Should().BeApproximately(0.881373587, 1e-6);
        InvokeTest<BFloat16>(1.0f, device, x => x.ACsch()).Should().BeApproximately(0.881373587f, 1e-3f);

        // Acsch(2) = 0.481211825
        InvokeTest<float>(2.0f, device, x => x.ACsch()).Should().BeApproximately(0.481211825f, 1e-6f);
        InvokeTest<double>(2.0, device, x => x.ACsch()).Should().BeApproximately(0.481211825, 1e-6);
        InvokeTest<BFloat16>(2.0f, device, x => x.ACsch()).Should().BeApproximately(0.48046875f, 1e-3f);

        // Acsch(0.5) = 1.443635475
        InvokeTest<float>(0.5f, device, x => x.ACsch()).Should().BeApproximately(1.443635475f, 1e-6f);
        InvokeTest<double>(0.5, device, x => x.ACsch()).Should().BeApproximately(1.443635475, 1e-6);
        InvokeTest<BFloat16>(0.5f, device, x => x.ACsch()).Should().BeApproximately(1.443635475f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForAcoth(IDevice device)
    {
        // Acoth(1) = ∞
        InvokeTest<float>(0.5f, device, x => x.ACoth()).Should().Be(float.NaN);
        InvokeTest<double>(0.5, device, x => x.ACoth()).Should().Be(double.NaN);
        InvokeTest<BFloat16>(0.5f, device, x => x.ACoth()).Should().Be(BFloat16.NaN);

        // Acoth(2) = 0.54930614433405484569762261846126
        InvokeTest<float>(2.0f, device, x => x.ACoth()).Should().BeApproximately(0.54930614433405484569762261846126f, 1e-6f);
        InvokeTest<double>(2.0, device, x => x.ACoth()).Should().BeApproximately(0.54930614433405484569762261846126, 1e-6);
        InvokeTest<BFloat16>(2.0f, device, x => x.ACoth()).Should().BeApproximately(0.54930614433405484569762261846126f, 1e-3f);

        // Acoth(-2) = -0.54930614433405484569762261846126
        InvokeTest<float>(-2.0f, device, x => x.ACoth()).Should().BeApproximately(-0.54930614433405484569762261846126f, 1e-6f);
        InvokeTest<double>(-2.0, device, x => x.ACoth()).Should().BeApproximately(-0.54930614433405484569762261846126, 1e-6);
        InvokeTest<BFloat16>(-2.0f, device, x => x.ACoth()).Should().BeApproximately(-0.54930614433405484569762261846126f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForSech2(IDevice device)
    {
        // Sech2(0) = 1
        InvokeTest<float>(0, device, x => x.Sech2()).Should().BeApproximately(1, 1e-6f);
        InvokeTest<double>(0, device, x => x.Sech2()).Should().BeApproximately(1, 1e-6);
        InvokeTest<BFloat16>(0, device, x => x.Sech2()).Should().BeApproximately(1, 1e-3f);

        // Sech2(1) = 0.4723665527
        InvokeTest<float>(1, device, x => x.Sech2()).Should().BeApproximately(0.4199743F, 1e-6f);
        InvokeTest<double>(1, device, x => x.Sech2()).Should().BeApproximately(0.4199743F, 1e-6);
        InvokeTest<BFloat16>(1, device, x => x.Sech2()).Should().BeApproximately(0.41796875f, 1e-3f);

        // Sech2(-1) = 0.4723665527
        InvokeTest<float>(-1, device, x => x.Sech2()).Should().BeApproximately(0.4199743F, 1e-6f);
        InvokeTest<double>(-1, device, x => x.Sech2()).Should().BeApproximately(0.4199743F, 1e-6);
        InvokeTest<BFloat16>(-1, device, x => x.Sech2()).Should().BeApproximately(0.41796875f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForCsch2(IDevice device)
    {
        // Csch2(0) = ∞
        InvokeTest<float>(0, device, x => x.Csch2()).Should().Be(float.PositiveInfinity);
        InvokeTest<double>(0, device, x => x.Csch2()).Should().Be(double.PositiveInfinity);
        InvokeTest<BFloat16>(0, device, x => x.Csch2()).Should().Be(BFloat16.PositiveInfinity);

        // Csch2(1) = 1.2752902
        InvokeTest<float>(1, device, x => x.Csch2()).Should().BeApproximately(0.7240616f, 1e-6f);
        InvokeTest<double>(1, device, x => x.Csch2()).Should().BeApproximately(0.7240616, 1e-6);
        InvokeTest<BFloat16>(1, device, x => x.Csch2()).Should().BeApproximately(0.7265625f, 1e-3f);

        // Csch2(-1) = 1.2752902
        InvokeTest<float>(-1, device, x => x.Csch2()).Should().BeApproximately(0.7240616f, 1e-6f);
        InvokeTest<double>(-1, device, x => x.Csch2()).Should().BeApproximately(0.7240616, 1e-6);
        InvokeTest<BFloat16>(-1, device, x => x.Csch2()).Should().BeApproximately(0.7265625f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForCoth2(IDevice device)
    {
        // Coth2(0) = ∞
        InvokeTest<float>(0, device, x => x.Coth2()).Should().Be(float.PositiveInfinity);
        InvokeTest<double>(0, device, x => x.Coth2()).Should().Be(double.PositiveInfinity);
        InvokeTest<BFloat16>(0, device, x => x.Coth2()).Should().Be(BFloat16.PositiveInfinity);

        // Coth2(1) = 1.3130352855
        InvokeTest<float>(1, device, x => x.Coth2()).Should().BeApproximately(1.7240616F, 1e-6f);
        InvokeTest<double>(1, device, x => x.Coth2()).Should().BeApproximately(1.7240616F, 1e-6);
        InvokeTest<BFloat16>(1, device, x => x.Coth2()).Should().BeApproximately(1.7421875f, 1e-2f);

        // Coth2(-1) = 1.3130352855
        InvokeTest<float>(-1, device, x => x.Coth2()).Should().BeApproximately(1.7240616F, 1e-6f);
        InvokeTest<double>(-1, device, x => x.Coth2()).Should().BeApproximately(1.7240616F, 1e-6);
        InvokeTest<BFloat16>(-1, device, x => x.Coth2()).Should().BeApproximately(1.7421875f, 1e-2f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForASec2(IDevice device)
    {
        // ASec2(-1) = -9.87
        InvokeTest<float>(-1, device, x => x.ASec2()).Should().BeApproximately(9.869605F, 1e-6f);
        InvokeTest<double>(-1, device, x => x.ASec2()).Should().BeApproximately(9.869605F, 1e-6);
        InvokeTest<BFloat16>(-1, device, x => x.ASec2()).Should().BeApproximately(9.869605F, 1e-3f);

        // ASec2(2) = 1.2309594
        InvokeTest<float>(1, device, x => x.ASec2()).Should().BeApproximately(0, 1e-6f);
        InvokeTest<double>(1, device, x => x.ASec2()).Should().BeApproximately(0, 1e-6);
        InvokeTest<BFloat16>(1, device, x => x.ASec2()).Should().BeApproximately(0, 1e-3f);

        // ASec2(2) = 1.0943951
        InvokeTest<float>(2, device, x => x.ASec2()).Should().BeApproximately(1.0966228F, 1e-6f);
        InvokeTest<double>(2, device, x => x.ASec2()).Should().BeApproximately(1.0966228F, 1e-6);
        InvokeTest<BFloat16>(2, device, x => x.ASec2()).Should().BeApproximately(1.0966228F, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForACsc2(IDevice device)
    {
        // ACsc2(-1) = -9.87
        InvokeTest<float>(-1, device, x => x.ACsc2()).Should().BeApproximately(2.4674013F, 1e-6f);
        InvokeTest<double>(-1, device, x => x.ACsc2()).Should().BeApproximately(2.4674013F, 1e-6);
        InvokeTest<BFloat16>(-1, device, x => x.ACsc2()).Should().BeApproximately(2.4674013F, 1e-3f);

        // ACsc2(1) = 0
        InvokeTest<float>(1, device, x => x.ACsc2()).Should().BeApproximately(2.4674013F, 1e-6f);
        InvokeTest<double>(1, device, x => x.ACsc2()).Should().BeApproximately(2.4674013F, 1e-6);
        InvokeTest<BFloat16>(1, device, x => x.ACsc2()).Should().BeApproximately(2.4674013F, 1e-3f);

        // ACsc2(2) = 1.2309594
        InvokeTest<float>(2, device, x => x.ACsc2()).Should().BeApproximately(0.2741557F, 1e-6f);
        InvokeTest<double>(2, device, x => x.ACsc2()).Should().BeApproximately(0.2741557F, 1e-6);
        InvokeTest<BFloat16>(2, device, x => x.ACsc2()).Should().BeApproximately(0.2741557F, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_ForACot2(IDevice device)
    {
        // ACot2(1) = π/4
        InvokeTest<float>(1, device, x => x.ACot2()).Should().BeApproximately(0.6168503F, 1e-6f);
        InvokeTest<double>(1, device, x => x.ACot2()).Should().BeApproximately(0.6168503F, 1e-6);
        InvokeTest<BFloat16>(1, device, x => x.ACot2()).Should().BeApproximately(0.6168503F, 1e-3f);

        // ACot2(0) = π/2
        InvokeTest<float>(0, device, x => x.ACot2()).Should().BeApproximately(2.4674013F, 1e-6f);
        InvokeTest<double>(0, device, x => x.ACot2()).Should().BeApproximately(2.4674013F, 1e-6);
        InvokeTest<BFloat16>(0, device, x => x.ACot2()).Should().BeApproximately(2.4674013F, 1e-3f);

        // ACot2(2) = 0.4636476
        InvokeTest<float>(2, device, x => x.ACot2()).Should().BeApproximately(0.2149691F, 1e-6f);
        InvokeTest<double>(2, device, x => x.ACot2()).Should().BeApproximately(0.2149691F, 1e-6);
        InvokeTest<BFloat16>(2, device, x => x.ACot2()).Should().BeApproximately(0.2149691F, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectResult_ForASech2(IDevice device)
    {
        // ASech2(0.5) = 1.7343782F
        InvokeTest<float>(0.5f, device, x => x.ASech2()).Should().BeApproximately(1.7343782F, 1e-6f);
        InvokeTest<double>(0.5, device, x => x.ASech2()).Should().BeApproximately(1.7343782F, 1e-6);
        InvokeTest<BFloat16>(0.5f, device, x => x.ASech2()).Should().BeApproximately(1.7421875f, 1e-3f);

        // ASech2(0.8) = 0.480453F
        InvokeTest<float>(0.8f, device, x => x.ASech2()).Should().BeApproximately(0.480453F, 1e-6f);
        InvokeTest<double>(0.8, device, x => x.ASech2()).Should().BeApproximately(0.480453F, 1e-6);
        InvokeTest<BFloat16>(0.8f, device, x => x.ASech2()).Should().BeApproximately(0.47851562f, 1e-3f);

        // ASech2(0.2) = 5.255243F
        InvokeTest<float>(0.2f, device, x => x.ASech2()).Should().BeApproximately(5.255243F, 1e-6f);
        InvokeTest<double>(0.2, device, x => x.ASech2()).Should().BeApproximately(5.255243F, 1e-6);
        InvokeTest<BFloat16>(0.2f, device, x => x.ASech2()).Should().BeApproximately(5.28125F, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectResult_ForACsch2(IDevice device)
    {
        // ACsch2(1) = 0.7768194F
        InvokeTest<float>(1.0f, device, x => x.ACsch2()).Should().BeApproximately(0.7768194F, 1e-6f);
        InvokeTest<double>(1.0, device, x => x.ACsch2()).Should().BeApproximately(0.7768194F, 1e-6);
        InvokeTest<BFloat16>(1.0f, device, x => x.ACsch2()).Should().BeApproximately(0.78125f, 1e-3f);

        // ACsch2(2) = 0.23156483F
        InvokeTest<float>(2.0f, device, x => x.ACsch2()).Should().BeApproximately(0.23156483F, 1e-6f);
        InvokeTest<double>(2.0, device, x => x.ACsch2()).Should().BeApproximately(0.23156483F, 1e-6);
        InvokeTest<BFloat16>(2.0f, device, x => x.ACsch2()).Should().BeApproximately(0.23046875f, 1e-3f);

        // ACsch2(0.5) = 2.0840833F
        InvokeTest<float>(0.5f, device, x => x.ACsch2()).Should().BeApproximately(2.0840833F, 1e-6f);
        InvokeTest<double>(0.5, device, x => x.ACsch2()).Should().BeApproximately(2.0840833F, 1e-6);
        InvokeTest<BFloat16>(0.5f, device, x => x.ACsch2()).Should().BeApproximately(2.09375f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectResult_ForACoth2(IDevice device)
    {
        // ACoth2(1) = ∞
        InvokeTest<float>(0.5f, device, x => x.ACoth2()).Should().Be(float.NaN);
        InvokeTest<double>(0.5, device, x => x.ACoth2()).Should().Be(double.NaN);
        InvokeTest<BFloat16>(0.5f, device, x => x.ACoth2()).Should().Be(BFloat16.NaN);

        // ACoth2(2) = 0.30173725F
        InvokeTest<float>(2.0f, device, x => x.ACoth2()).Should().BeApproximately(0.30173725F, 1e-6f);
        InvokeTest<double>(2.0, device, x => x.ACoth2()).Should().BeApproximately(0.30173725F, 1e-6);
        InvokeTest<BFloat16>(2.0f, device, x => x.ACoth2()).Should().BeApproximately(0.30273438f, 1e-3f);

        // ACoth2(-2) = -0.30173725F
        InvokeTest<float>(-2.0f, device, x => x.ACoth2()).Should().BeApproximately(0.30173725F, 1e-6f);
        InvokeTest<double>(-2.0, device, x => x.ACoth2()).Should().BeApproximately(0.30173725F, 1e-6);
        InvokeTest<BFloat16>(-2.0f, device, x => x.ACoth2()).Should().BeApproximately(0.30273438f, 1e-3f);
    }

    private static TNumber InvokeTest<TNumber>(TNumber value, IDevice device, Func<Scalar<TNumber>, Scalar<TNumber>> function)
        where TNumber : unmanaged, ITrigonometricFunctions<TNumber>, INumber<TNumber>
    {
        var tensor = new Scalar<TNumber>(value);
        tensor.To(device);

        return function(tensor).Value;
    }
}