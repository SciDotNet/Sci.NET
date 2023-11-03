// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using Sci.NET.Common;

namespace Sci.NET.CUDA.RuntimeApi.Types;

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
internal readonly struct CudaDeviceProps : IEquatable<CudaDeviceProps>
{
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
    public readonly char[] _name;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
    public readonly byte[] _uuid;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
    public readonly byte[] _luid;

    public readonly uint _luidDeviceNodeMask;
    public readonly SizeT _totalGlobalMem;
    public readonly SizeT _sharedMemPerBlock;
    public readonly int _regsPerBlock;
    public readonly int _warpSize;
    public readonly SizeT _memPitch;
    public readonly int _maxThreadsPerBlock;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
    public readonly int[] _maxThreadsDim;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
    public readonly int[] _maxGridSize;

    public readonly int _clockRate;
    public readonly SizeT _totalConstMem;
    public readonly int _major;
    public readonly int _minor;
    public readonly SizeT _textureAlignment;
    public readonly SizeT _texturePitchAlignment;
    public readonly int _deviceOverlap;
    public readonly int _multiProcessorCount;
    public readonly int _kernelExecTimeoutEnabled;
    public readonly int _integrated;
    public readonly int _canMapHostMemory;
    public readonly int _computeMode;
    public readonly int _maxTexture1D;
    public readonly int _maxTexture1DMipmap;
    public readonly int _maxTexture1DLinear;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
    public readonly int[] _maxTexture2D;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
    public readonly int[] _maxTexture2DMipmap;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
    public readonly int[] _maxTexture2DLinear;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
    public readonly int[] _maxTexture2DGather;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
    public readonly int[] _maxTexture3D;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
    public readonly int[] _maxTexture3DAlt;

    public readonly int _maxTextureCubemap;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
    public readonly int[] _maxTexture1DLayered;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
    public readonly int[] _maxTexture2DLayered;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
    public readonly int[] _maxTextureCubemapLayered;

    public readonly int _maxSurface1D;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
    public readonly int[] _maxSurface2D;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
    public readonly int[] _maxSurface3D;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
    public readonly int[] _maxSurface1DLayered;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
    public readonly int[] _maxSurface2DLayered;

    public readonly int _maxSurfaceCubemap;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
    public readonly int[] _maxSurfaceCubemapLayered;

    public readonly SizeT _surfaceAlignment;
    public readonly int _concurrentKernels;
    public readonly int _ECCEnabled;
    public readonly int _pciBusID;
    public readonly int _pciDeviceID;
    public readonly int _pciDomainID;
    public readonly int _tccDriver;
    public readonly int _asyncEngineCount;
    public readonly int _unifiedAddressing;
    public readonly int _memoryClockRate;
    public readonly int _memoryBusWidth;
    public readonly int _l2CacheSize;
    public readonly int _persistingL2CacheMaxSize;
    public readonly int _maxThreadsPerMultiProcessor;
    public readonly int _streamPrioritiesSupported;
    public readonly int _globalL1CacheSupported;
    public readonly int _localL1CacheSupported;
    public readonly SizeT _sharedMemPerMultiprocessor;
    public readonly int _regsPerMultiprocessor;
    public readonly int _managedMemory;
    public readonly int _isMultiGpuBoard;
    public readonly int _multiGpuBoardGroupID;
    public readonly int _hostNativeAtomicSupported;
    public readonly int _singleToDoublePrecisionPerfRatio;
    public readonly int _pageableMemoryAccess;
    public readonly int _concurrentManagedAccess;
    public readonly int _computePreemptionSupported;
    public readonly int _canUseHostPointerForRegisteredMem;
    public readonly int _cooperativeLaunch;
    public readonly int _cooperativeMultiDeviceLaunch;
    public readonly SizeT _sharedMemPerBlockOptin;
    public readonly int _pageableMemoryAccessUsesHostPageTables;
    public readonly int _directManagedMemAccessFromHost;
    public readonly int _maxBlocksPerMultiProcessor;
    public readonly int _accessPolicyMaxWindowSize;
    public readonly SizeT _reservedSharedMemPerBlock;

    public static bool operator ==(CudaDeviceProps left, CudaDeviceProps right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(CudaDeviceProps left, CudaDeviceProps right)
    {
        return !left.Equals(right);
    }

#pragma warning disable CA1502
    public bool Equals(CudaDeviceProps other)
    {
        return _name.Equals(other._name) &&
               _uuid.Equals(other._uuid) &&
               _luid.Equals(other._luid) &&
               _luidDeviceNodeMask == other._luidDeviceNodeMask &&
               _totalGlobalMem.Equals(other._totalGlobalMem) &&
               _sharedMemPerBlock.Equals(other._sharedMemPerBlock) &&
               _regsPerBlock == other._regsPerBlock &&
               _warpSize == other._warpSize &&
               _memPitch.Equals(other._memPitch) &&
               _maxThreadsPerBlock == other._maxThreadsPerBlock &&
               _maxThreadsDim.Equals(other._maxThreadsDim) &&
               _maxGridSize.Equals(other._maxGridSize) &&
               _clockRate == other._clockRate &&
               _totalConstMem.Equals(other._totalConstMem) &&
               _major == other._major &&
               _minor == other._minor &&
               _textureAlignment.Equals(other._textureAlignment) &&
               _texturePitchAlignment.Equals(other._texturePitchAlignment) &&
               _deviceOverlap == other._deviceOverlap &&
               _multiProcessorCount == other._multiProcessorCount &&
               _kernelExecTimeoutEnabled == other._kernelExecTimeoutEnabled &&
               _integrated == other._integrated &&
               _canMapHostMemory == other._canMapHostMemory &&
               _computeMode == other._computeMode &&
               _maxTexture1D == other._maxTexture1D &&
               _maxTexture1DMipmap == other._maxTexture1DMipmap &&
               _maxTexture1DLinear == other._maxTexture1DLinear &&
               _maxTexture2D.Equals(other._maxTexture2D) &&
               _maxTexture2DMipmap.Equals(other._maxTexture2DMipmap) &&
               _maxTexture2DLinear.Equals(other._maxTexture2DLinear) &&
               _maxTexture2DGather.Equals(other._maxTexture2DGather) &&
               _maxTexture3D.Equals(other._maxTexture3D) &&
               _maxTexture3DAlt.Equals(other._maxTexture3DAlt) &&
               _maxTextureCubemap == other._maxTextureCubemap &&
               _maxTexture1DLayered.Equals(other._maxTexture1DLayered) &&
               _maxTexture2DLayered.Equals(other._maxTexture2DLayered) &&
               _maxTextureCubemapLayered.Equals(other._maxTextureCubemapLayered) &&
               _maxSurface1D == other._maxSurface1D &&
               _maxSurface2D.Equals(other._maxSurface2D) &&
               _maxSurface3D.Equals(other._maxSurface3D) &&
               _maxSurface1DLayered.Equals(other._maxSurface1DLayered) &&
               _maxSurface2DLayered.Equals(other._maxSurface2DLayered) &&
               _maxSurfaceCubemap == other._maxSurfaceCubemap &&
               _maxSurfaceCubemapLayered.Equals(other._maxSurfaceCubemapLayered) &&
               _surfaceAlignment.Equals(other._surfaceAlignment) &&
               _concurrentKernels == other._concurrentKernels &&
               _ECCEnabled == other._ECCEnabled &&
               _pciBusID == other._pciBusID &&
               _pciDeviceID == other._pciDeviceID &&
               _pciDomainID == other._pciDomainID &&
               _tccDriver == other._tccDriver &&
               _asyncEngineCount == other._asyncEngineCount &&
               _unifiedAddressing == other._unifiedAddressing &&
               _memoryClockRate == other._memoryClockRate &&
               _memoryBusWidth == other._memoryBusWidth &&
               _l2CacheSize == other._l2CacheSize &&
               _persistingL2CacheMaxSize == other._persistingL2CacheMaxSize &&
               _maxThreadsPerMultiProcessor == other._maxThreadsPerMultiProcessor &&
               _streamPrioritiesSupported == other._streamPrioritiesSupported &&
               _globalL1CacheSupported == other._globalL1CacheSupported &&
               _localL1CacheSupported == other._localL1CacheSupported &&
               _sharedMemPerMultiprocessor.Equals(other._sharedMemPerMultiprocessor) &&
               _regsPerMultiprocessor == other._regsPerMultiprocessor &&
               _managedMemory == other._managedMemory &&
               _isMultiGpuBoard == other._isMultiGpuBoard &&
               _multiGpuBoardGroupID == other._multiGpuBoardGroupID &&
               _hostNativeAtomicSupported == other._hostNativeAtomicSupported &&
               _singleToDoublePrecisionPerfRatio == other._singleToDoublePrecisionPerfRatio &&
               _pageableMemoryAccess == other._pageableMemoryAccess &&
               _concurrentManagedAccess == other._concurrentManagedAccess &&
               _computePreemptionSupported == other._computePreemptionSupported &&
               _canUseHostPointerForRegisteredMem == other._canUseHostPointerForRegisteredMem &&
               _cooperativeLaunch == other._cooperativeLaunch &&
               _cooperativeMultiDeviceLaunch == other._cooperativeMultiDeviceLaunch &&
               _sharedMemPerBlockOptin.Equals(other._sharedMemPerBlockOptin) &&
               _pageableMemoryAccessUsesHostPageTables == other._pageableMemoryAccessUsesHostPageTables &&
               _directManagedMemAccessFromHost == other._directManagedMemAccessFromHost &&
               _maxBlocksPerMultiProcessor == other._maxBlocksPerMultiProcessor &&
               _accessPolicyMaxWindowSize == other._accessPolicyMaxWindowSize &&
               _reservedSharedMemPerBlock.Equals(other._reservedSharedMemPerBlock);
    }
#pragma warning restore CA1502

    public override bool Equals(object? obj)
    {
        return obj is CudaDeviceProps other && Equals(other);
    }

    public override int GetHashCode()
    {
        var hashCode = default(HashCode);
        hashCode.Add(_name);
        hashCode.Add(_uuid);
        hashCode.Add(_luid);
        hashCode.Add(_luidDeviceNodeMask);
        hashCode.Add(_totalGlobalMem);
        hashCode.Add(_sharedMemPerBlock);
        hashCode.Add(_regsPerBlock);
        hashCode.Add(_warpSize);
        hashCode.Add(_memPitch);
        hashCode.Add(_maxThreadsPerBlock);
        hashCode.Add(_maxThreadsDim);
        hashCode.Add(_maxGridSize);
        hashCode.Add(_clockRate);
        hashCode.Add(_totalConstMem);
        hashCode.Add(_major);
        hashCode.Add(_minor);
        hashCode.Add(_textureAlignment);
        hashCode.Add(_texturePitchAlignment);
        hashCode.Add(_deviceOverlap);
        hashCode.Add(_multiProcessorCount);
        hashCode.Add(_kernelExecTimeoutEnabled);
        hashCode.Add(_integrated);
        hashCode.Add(_canMapHostMemory);
        hashCode.Add(_computeMode);
        hashCode.Add(_maxTexture1D);
        hashCode.Add(_maxTexture1DMipmap);
        hashCode.Add(_maxTexture1DLinear);
        hashCode.Add(_maxTexture2D);
        hashCode.Add(_maxTexture2DMipmap);
        hashCode.Add(_maxTexture2DLinear);
        hashCode.Add(_maxTexture2DGather);
        hashCode.Add(_maxTexture3D);
        hashCode.Add(_maxTexture3DAlt);
        hashCode.Add(_maxTextureCubemap);
        hashCode.Add(_maxTexture1DLayered);
        hashCode.Add(_maxTexture2DLayered);
        hashCode.Add(_maxTextureCubemapLayered);
        hashCode.Add(_maxSurface1D);
        hashCode.Add(_maxSurface2D);
        hashCode.Add(_maxSurface3D);
        hashCode.Add(_maxSurface1DLayered);
        hashCode.Add(_maxSurface2DLayered);
        hashCode.Add(_maxSurfaceCubemap);
        hashCode.Add(_maxSurfaceCubemapLayered);
        hashCode.Add(_surfaceAlignment);
        hashCode.Add(_concurrentKernels);
        hashCode.Add(_ECCEnabled);
        hashCode.Add(_pciBusID);
        hashCode.Add(_pciDeviceID);
        hashCode.Add(_pciDomainID);
        hashCode.Add(_tccDriver);
        hashCode.Add(_asyncEngineCount);
        hashCode.Add(_unifiedAddressing);
        hashCode.Add(_memoryClockRate);
        hashCode.Add(_memoryBusWidth);
        hashCode.Add(_l2CacheSize);
        hashCode.Add(_persistingL2CacheMaxSize);
        hashCode.Add(_maxThreadsPerMultiProcessor);
        hashCode.Add(_streamPrioritiesSupported);
        hashCode.Add(_globalL1CacheSupported);
        hashCode.Add(_localL1CacheSupported);
        hashCode.Add(_sharedMemPerMultiprocessor);
        hashCode.Add(_regsPerMultiprocessor);
        hashCode.Add(_managedMemory);
        hashCode.Add(_isMultiGpuBoard);
        hashCode.Add(_multiGpuBoardGroupID);
        hashCode.Add(_hostNativeAtomicSupported);
        hashCode.Add(_singleToDoublePrecisionPerfRatio);
        hashCode.Add(_pageableMemoryAccess);
        hashCode.Add(_concurrentManagedAccess);
        hashCode.Add(_computePreemptionSupported);
        hashCode.Add(_canUseHostPointerForRegisteredMem);
        hashCode.Add(_cooperativeLaunch);
        hashCode.Add(_cooperativeMultiDeviceLaunch);
        hashCode.Add(_sharedMemPerBlockOptin);
        hashCode.Add(_pageableMemoryAccessUsesHostPageTables);
        hashCode.Add(_directManagedMemAccessFromHost);
        hashCode.Add(_maxBlocksPerMultiProcessor);
        hashCode.Add(_accessPolicyMaxWindowSize);
        hashCode.Add(_reservedSharedMemPerBlock);
        return hashCode.ToHashCode();
    }
}