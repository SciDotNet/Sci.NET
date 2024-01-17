// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.Backends.Devices;

/// <summary>
/// An interface for a computation device.
/// </summary>
[PublicAPI]
public interface IDevice : IEquatable<IDevice>
{
    /// <summary>
    /// Gets the Id of the current device.
    /// </summary>
    public Guid Id { get; }

    /// <summary>
    /// Gets the name of the device.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the <see cref="ITensorStorageKernels"/> of the device.
    /// </summary>
    public ITensorStorageKernels Storage { get; }

    /// <summary>
    /// Gets the category of the device.
    /// </summary>
    public DeviceCategory Category { get; }

    /// <summary>
    /// Gets the <see cref="ITensorBackend"/> of the device.
    /// </summary>
    /// <returns>The <see cref="ITensorBackend"/> for the device.</returns>
    public ITensorBackend GetTensorBackend();
}