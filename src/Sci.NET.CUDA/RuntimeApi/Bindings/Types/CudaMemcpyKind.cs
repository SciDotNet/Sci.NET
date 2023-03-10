// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Types;

/// <summary>
/// Enumeration of CUDA Memory Copy kinds.
/// </summary>
[PublicAPI]
public enum CudaMemcpyKind
{
    /// <summary>
    /// Host to host memory copy.
    /// </summary>
    HostToHost = 0,

    /// <summary>
    /// Host to device memory copy.
    /// </summary>
    HostToDevice = 1,

    /// <summary>
    /// Device to host memory copy.
    /// </summary>
    DeviceToHost = 2,

    /// <summary>
    /// Device to device memory copy.
    /// </summary>
    DeviceToDevice = 3,

    /// <summary>
    /// Default memory copy type.
    /// </summary>
    Default = 4,
}
