// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.CUDA.RuntimeApi.Types;

namespace Sci.NET.CUDA;

/// <summary>
/// Represents the properties of a CUDA device.
/// </summary>
[PublicAPI]
public class CudaDeviceProperties
{
#pragma warning disable SA1206

    /// <summary>
    /// Gets the name of the device.
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Gets the major compute capability of the device.
    /// </summary>
    public required int ComputeCapabilityMajor { get; init; }

    /// <summary>
    /// Gets the minor compute capability of the device.
    /// </summary>
    public required int ComputeCapabilityMinor { get; init; }

    /// <summary>
    /// Gets the index of the device.
    /// </summary>
    public required int Index { get; init; }

    /// <summary>
    /// Gets the maximum number of threads per block.
    /// </summary>
    public required int MaxThreadsPerBlock { get; init; }

    /// <summary>
    /// Gets the maximum number of threads per multiprocessor.
    /// </summary>
    public required int MaxThreadsPerMultiProcessor { get; init; }

    /// <summary>
    /// Gets the maximum number of warps per multiprocessor.
    /// </summary>
    public required int MaxGridSizeX { get; init; }

    /// <summary>
    /// Gets the maximum number of blocks per multiprocessor.
    /// </summary>
    public required int MaxGridSizeY { get; init; }

    /// <summary>
    /// Gets the maximum number of blocks per multiprocessor.
    /// </summary>
    public required int MaxGridSizeZ { get; init; }

    /// <summary>
    /// Gets the maximum amount of shared memory per block.
    /// </summary>
    public required long MaxSharedMemoryPerBlock { get; init; }

    /// <summary>
    /// Gets the maximum number of registers per block.
    /// </summary>
    public required int MaxRegistersPerBlock { get; init; }

    /// <summary>
    /// Gets the warp size.
    /// </summary>
    public required int WarpSize { get; init; }

    /// <summary>
    /// Gets the maximum pitch in bytes allowed by the memory copy functions that involve memory regions allocated through cudaMallocPitch().
    /// </summary>
    public required long MaxPitch { get; init; }

    /// <summary>
    /// Gets the memory clock rate in kilohertz.
    /// </summary>
    public required long MemoryClockRate { get; init; }

    /// <summary>
    /// Gets the total amount of global memory available on the device in bytes.
    /// </summary>
    public required long TotalGlobalMemory { get; init; }

#pragma warning restore SA1206

    internal static CudaDeviceProperties FromNativeProps(CudaDeviceProps props, int index)
    {
        return new CudaDeviceProperties
        {
            Name = new string(props._name).TrimEnd('\0'),
            ComputeCapabilityMajor = props._major,
            ComputeCapabilityMinor = props._minor,
            Index = index,
            MaxThreadsPerBlock = props._maxThreadsPerBlock,
            MaxThreadsPerMultiProcessor = props._maxThreadsPerMultiProcessor,
            MaxGridSizeX = props._maxGridSize[0],
            MaxGridSizeY = props._maxGridSize[1],
            MaxGridSizeZ = props._maxGridSize[2],
            MaxSharedMemoryPerBlock = props._sharedMemPerBlock.ToInt64(),
            MaxRegistersPerBlock = props._regsPerBlock,
            WarpSize = props._warpSize,
            MaxPitch = props._memPitch.ToInt64(),
            MemoryClockRate = props._clockRate,
            TotalGlobalMemory = props._totalGlobalMem.ToInt64()
        };
    }
}