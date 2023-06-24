// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.Backends.Devices;

/// <summary>
/// Enumerates device categories.
/// </summary>
[PublicAPI]
public enum DeviceCategory
{
    /// <summary>
    /// A CPU device.
    /// </summary>
    Cpu = 0,

    /// <summary>
    /// A GPU device.
    /// </summary>
    Gpu = 1,

    /// <summary>
    /// A TPU device.
    /// </summary>
    Tpu = 2,

    /// <summary>
    /// An FPGA device.
    /// </summary>
    Fpga = 3,

    /// <summary>
    /// An ASIC device.
    /// </summary>
    Asic = 4,

    /// <summary>
    /// A CUDA device.
    /// </summary>
    Cuda = 5
}