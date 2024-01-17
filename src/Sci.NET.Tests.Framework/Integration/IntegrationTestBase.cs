// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Tests.Framework.Integration;

/// <summary>
/// Base class for integration tests.
/// </summary>
public abstract class IntegrationTestBase
{
    /// <summary>
    /// Gets the devices to use for integration tests.
    /// </summary>
    public static TheoryData<IDevice> ComputeDevices => new () { new CpuComputeDevice() };
}