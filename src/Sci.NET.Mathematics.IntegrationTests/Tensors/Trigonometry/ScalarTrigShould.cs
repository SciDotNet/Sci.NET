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
        Sin<float>(0.0f, device).Should().BeApproximately(0.0f, 1e-6f);
        Sin<double>(0.0, device).Should().BeApproximately(0.0, 1e-6);
        Sin<BFloat16>(0.0f, device).Should().BeApproximately(0.0f, 1e-3f);

        // Sin(π/2) = 1
        Sin<float>(MathF.PI / 2.0f, device).Should().BeApproximately(1.0f, 1e-6f);
        Sin<double>(Math.PI / 2.0, device).Should().BeApproximately(1.0, 1e-6);
        Sin<BFloat16>(MathF.PI / 2.0f, device).Should().BeApproximately(1.0f, 1e-3f);

        // Sin(π) = 0
        Sin<float>(MathF.PI, device).Should().BeApproximately(0.0f, 1e-6f);
        Sin<double>(Math.PI, device).Should().BeApproximately(0.0, 1e-6);
        Sin<BFloat16>(MathF.PI, device).Should().BeApproximately(0.0f, 1e-3f);

        // Sin(3π/2) = -1
        Sin<float>(3.0f * MathF.PI / 2.0f, device).Should().BeApproximately(-1.0f, 1e-6f);
        Sin<double>(3.0 * Math.PI / 2.0, device).Should().BeApproximately(-1.0, 1e-6);
        Sin<BFloat16>(3.0f * MathF.PI / 2.0f, device).Should().BeApproximately(-1.0f, 1e-3f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectResult_ForCos(IDevice device)
    {
        // Cos(0) = 1
        Cos<float>(0.0f, device).Should().BeApproximately(1.0f, 1e-6f);
        Cos<double>(0.0, device).Should().BeApproximately(1.0, 1e-6);
        Cos<BFloat16>(0.0f, device).Should().BeApproximately(1.0f, 1e-3f);

        // Cos(π/2) = 0
        Cos<float>(MathF.PI / 2.0f, device).Should().BeApproximately(0.0f, 1e-6f);
        Cos<double>(Math.PI / 2.0, device).Should().BeApproximately(0.0, 1e-6);
        Cos<BFloat16>(MathF.PI / 2.0f, device).Should().BeApproximately(0.0f, 1e-3f);

        // Cos(π) = -1
        Cos<float>(MathF.PI, device).Should().BeApproximately(-1.0f, 1e-6f);
        Cos<double>(Math.PI, device).Should().BeApproximately(-1.0, 1e-6);
        Cos<BFloat16>(MathF.PI, device).Should().BeApproximately(-1.0f, 1e-3f);

        // Cos(3π/2) = 0
        Cos<float>(3.0f * MathF.PI / 2.0f, device).Should().BeApproximately(0.0f, 1e-6f);
        Cos<double>(3.0 * Math.PI / 2.0, device).Should().BeApproximately(0.0, 1e-6);
        Cos<BFloat16>(3.0f * MathF.PI / 2.0f, device).Should().BeApproximately(0.0f, 1e-2f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectResult_ForTan(IDevice device)
    {
        // Tan(0) = 0
        Tan<float>(0.0f, device).Should().BeApproximately(0.0f, 1e-6f);
        Tan<double>(0.0, device).Should().BeApproximately(0.0, 1e-6);
        Tan<BFloat16>(0.0f, device).Should().BeApproximately(0.0f, 1e-3f);

        // Tan(π/4) = 1
        Tan<float>(MathF.PI / 4.0f, device).Should().BeApproximately(1.0f, 1e-6f);
        Tan<double>(Math.PI / 4.0, device).Should().BeApproximately(1.0, 1e-6);
        Tan<BFloat16>(MathF.PI / 4.0f, device).Should().BeApproximately(1.0f, 1e-3f);

        // Tan(3π/4) = -1
        Tan<float>(3.0f * MathF.PI / 4.0f, device).Should().BeApproximately(-1.0f, 1e-6f);
        Tan<double>(3.0 * Math.PI / 4.0, device).Should().BeApproximately(-1.0, 1e-6);
        Tan<BFloat16>(3.0f * MathF.PI / 4.0f, device).Should().BeApproximately(-1.0f, 1e-2f);
    }

    private static TNumber Sin<TNumber>(TNumber value, IDevice device)
        where TNumber : unmanaged, ITrigonometricFunctions<TNumber>, INumber<TNumber>
    {
        var tensor = new Scalar<TNumber>(value);
        tensor.To(device);

        return tensor.Sin().Value;
    }

    private static TNumber Cos<TNumber>(TNumber value, IDevice device)
        where TNumber : unmanaged, ITrigonometricFunctions<TNumber>, INumber<TNumber>
    {
        var tensor = new Scalar<TNumber>(value);
        tensor.To(device);

        return tensor.Cos().Value;
    }

    private static TNumber Tan<TNumber>(TNumber value, IDevice device)
        where TNumber : unmanaged, ITrigonometricFunctions<TNumber>, INumber<TNumber>
    {
        var tensor = new Scalar<TNumber>(value);
        tensor.To(device);

        return tensor.Tan().Value;
    }
}