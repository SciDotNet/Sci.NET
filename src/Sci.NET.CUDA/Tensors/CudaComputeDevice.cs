// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.CUDA.Native;
using Sci.NET.CUDA.Tensors.Backend;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.CUDA.Tensors;

/// <summary>
/// Represents a CUDA device.
/// </summary>
[PublicAPI]
public class CudaComputeDevice : IDevice
{
    private static readonly CudaComputeDevice[] Devices;

#pragma warning disable CA1810
    static CudaComputeDevice()
#pragma warning restore CA1810
    {
        Devices = new CudaComputeDevice[NativeApi.CudaGetDeviceCount()];

        for (var i = 0; i < Devices.Length; i++)
        {
            var props = NativeApi.GetDeviceProperties(i);
            Devices[i] = new CudaComputeDevice(props);
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaComputeDevice"/> class.
    /// </summary>
    /// <param name="deviceIdx">The index of the device.</param>
    public CudaComputeDevice(int deviceIdx)
    {
        Properties = Devices[deviceIdx].Properties;
        Id = Devices[deviceIdx].Id;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaComputeDevice"/> class.
    /// </summary>
    public CudaComputeDevice()
        : this(0)
    {
    }

    private CudaComputeDevice(CudaDeviceProperties props)
    {
        Properties = props;
        Id = Guid.NewGuid();
    }

    /// <summary>
    /// Gets the device properties.
    /// </summary>
    public CudaDeviceProperties Properties { get; }

    /// <inheritdoc />
    public Guid Id { get; }

    /// <inheritdoc />
    public string Name => Properties.Name;

    /// <inheritdoc />
    public DeviceCategory Category => DeviceCategory.Cuda;

    /// <inheritdoc />
    public bool Equals(IDevice? other)
    {
        return other is CudaComputeDevice device && device.Id == Id;
    }

    /// <inheritdoc />
    public ITensorBackend GetTensorBackend()
    {
        return new CudaTensorBackend();
    }
}