// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics;
using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Tests.Framework.Integration;

/// <summary>
/// Base class for integration tests.
/// </summary>
#pragma warning disable CA1515
public abstract class IntegrationTestBase
#pragma warning restore CA1515
{
    /// <summary>
    /// Initializes a new instance of the <see cref="IntegrationTestBase"/> class.
    /// </summary>
    protected IntegrationTestBase()
    {
        // Ensure that the preview features are enabled for the tests.
        SciDotNetConfiguration.PreviewFeatures.EnableAutoGrad();
    }

    /// <summary>
    /// Gets the devices to use for integration tests.
    /// </summary>
    public static TheoryData<IDevice> ComputeDevices => new() { new CpuComputeDevice() };
}