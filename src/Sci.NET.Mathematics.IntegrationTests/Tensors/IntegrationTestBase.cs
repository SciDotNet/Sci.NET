// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors;

public abstract class IntegrationTestBase
{
    public static TheoryData<IDevice> ComputeDevices => new () { new CpuComputeDevice() };
}