// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Runtime;
using Sci.NET.Mathematics.Backends.Managed;

namespace Sci.NET.Mathematics.Backends.Devices;

/// <summary>
/// A CPU compute device.
/// </summary>
[PublicAPI]
public class CpuComputeDevice : IDevice
{
    private static readonly CpuComputeDevice Instance = new(Guid.NewGuid(), CpuInfo.GetInfoString());

    /// <summary>
    /// Initializes a new instance of the <see cref="CpuComputeDevice"/> class.
    /// </summary>
    public CpuComputeDevice()
    {
        Id = Instance.Id;
        Name = Instance.Name;
        Storage = Instance.Storage;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CpuComputeDevice"/> class.
    /// </summary>
    /// <param name="id">The Id of the device.</param>
    /// <param name="name">The name of the device.</param>
    public CpuComputeDevice(Guid id, string name)
    {
        Name = name;
        Id = id;
        Storage = new ManagedStorageKernels();
    }

    /// <inheritdoc />
    public Guid Id { get; }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public ITensorStorageKernels Storage { get; }

    /// <inheritdoc />
    public DeviceCategory Category => DeviceCategory.Cpu;

    /// <inheritdoc />
    public ITensorBackend GetTensorBackend()
    {
        return ManagedTensorBackend.Instance;
    }

    /// <inheritdoc />
    public bool Equals(IDevice? other)
    {
        return other is not null && Id == other.Id;
    }
}