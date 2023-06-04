// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.Runtime.Structs;

/// <summary>
/// Enumerates the status codes returned by the CUDA Runtime API.
/// </summary>
public enum CudaStatusCode
{
     /// <summary>
    /// The API call returned with no errors. In the case of query calls, this also means that the operation being queried
    /// is complete (see cudaEventQuery() and cudaStreamQuery()).
    /// </summary>
    CudaSuccess = 0,

    /// <summary>
    /// This indicates that one or more of the parameters passed to the API call is not within an acceptable range of
    /// values.
    /// </summary>
    CudaErrorInvalidValue = 1,

    /// <summary>
    /// The API call failed because it was unable to sdn_allocate enough memory to perform the requested operation.
    /// </summary>
    CudaErrorMemoryAllocation = 2,

    /// <summary>
    /// The API call failed because the CUDA driver and runtime could not be initialized.
    /// </summary>
    CudaErrorInitializationError = 3,

    /// <summary>
    /// This indicates that a CUDA Runtime API call cannot be executed because it is being called during process shut down,
    /// at a point in time after CUDA driver has been unloaded.
    /// </summary>
    CudaErrorCudartUnloading = 4,

    /// <summary>
    /// This indicates profiler is not initialized for this run. This can happen when the application is running with
    /// external profiling tools like visual profiler.
    /// </summary>
    CudaErrorProfilerDisabled = 5,

    /// <summary>
    /// This indicates that a kernel launch is requesting resources that can never be satisfied by the current device.
    /// Requesting more shared memory per block than the device supports will trigger this error, as will requesting too
    /// many threads or blocks. See cudaDeviceProp for more device limitations.
    /// </summary>
    CudaErrorInvalidConfiguration = 9,

    /// <summary>
    /// This indicates that one or more of the pitch-related parameters passed to the API call is not within the acceptable
    /// range for pitch.
    /// </summary>
    CudaErrorInvalidPitchValue = 12,

    /// <summary>
    /// This indicates that the symbol name/identifier passed to the API call is not a valid name or identifier.
    /// </summary>
    CudaErrorInvalidSymbol = 13,

    /// <summary>
    /// This indicates that at least one device pointer passed to the API call is not a valid device pointer.
    /// </summary>
    CudaErrorInvalidDevicePointer = 17,

    /// <summary>
    /// This indicates that the texture passed to the API call is not a valid texture.
    /// </summary>
    CudaErrorInvalidTexture = 18,

    /// <summary>
    /// This indicates that the texture binding is not valid. This occurs if you call cudaGetTextureAlignmentOffset() with
    /// an unbound texture.
    /// </summary>
    CudaErrorInvalidTextureBinding = 19,

    /// <summary>
    /// This indicates that the channel descriptor passed to the API call is not valid. This occurs if the format is not
    /// one of the formats specified by cudaChannelFormatKind, or if one of the dimensions is invalid.
    /// </summary>
    CudaErrorInvalidChannelDescriptor = 20,

    /// <summary>
    /// This indicates that the direction of the memcpy passed to the API call is not one of the types specified by
    /// cudaMemcpyKind.
    /// </summary>
    CudaErrorInvalidMemcpyDirection = 21,

    /// <summary>
    /// This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA.
    /// </summary>
    CudaErrorInvalidFilterSetting = 26,

    /// <summary>
    /// This indicates that an attempt was made to read a non-float texture as a normalized float. This is not supported by
    /// CUDA.
    /// </summary>
    CudaErrorInvalidNormSetting = 27,

    /// <summary>
    /// This indicates that the CUDA driver that the application has loaded is a stub library. Applications that run with
    /// the stub rather than a real driver loaded will result in CUDA API returning this error.
    /// </summary>
    CudaErrorStubLibrary = 34,

    /// <summary>
    /// This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library. This is not a
    /// supported configuration. Users should install an updated NVIDIA display driver to allow the application to run.
    /// </summary>
    CudaErrorInsufficientDriver = 35,

    /// <summary>
    /// This indicates that the API call requires a newer CUDA driver than the one currently installed. Users should
    /// install an updated NVIDIA CUDA driver to allow the API call to succeed.
    /// </summary>
    CudaErrorCallRequiresNewerDriver = 36,

    /// <summary>
    /// This indicates that the surface passed to the API call is not a valid surface.
    /// </summary>
    CudaErrorInvalidSurface = 37,

    /// <summary>
    /// This indicates that multiple global or constant variables (across separate CUDA source files in the application)
    /// share the same string name.
    /// </summary>
    CudaErrorDuplicateVariableName = 43,

    /// <summary>
    /// This indicates that multiple textures (across separate CUDA source files in the application) share the same string
    /// name.
    /// </summary>
    CudaErrorDuplicateTextureName = 44,

    /// <summary>
    /// This indicates that multiple surfaces (across separate CUDA source files in the application) share the same string
    /// name.
    /// </summary>
    CudaErrorDuplicateSurfaceName = 45,

    /// <summary>
    /// This indicates that all CUDA devices are busy or unavailable at the current time. Devices are often
    /// busy/unavailable due to use of cudaComputeModeExclusive, cudaComputeModeProhibited or when long running CUDA
    /// kernels have filled up the GPU and are blocking new work from starting. They can also be unavailable due to memory
    /// constraints on a device that already has active CUDA work being performed.
    /// </summary>
    CudaErrorDevicesUnavailable = 46,

    /// <summary>
    /// This indicates that the current context is not compatible with this the CUDA Runtime. This can only occur if you
    /// are using CUDA Runtime/Driver interoperability and have created an existing Driver context using the driver API.
    /// The Driver context may be incompatible either because the Driver context was created using an older version of the
    /// API, because the Runtime API call expects a primary driver context and the Driver context is not primary, or
    /// because the Driver context has been destroyed. Please see Interactions with the CUDA Driver API" for more
    /// information.
    /// </summary>
    CudaErrorIncompatibleDriverContext = 49,

    /// <summary>
    /// The device function being invoked (usually via cudaLaunchKernel()) was not previously configured via the
    /// cudaConfigureCall() function.
    /// </summary>
    CudaErrorMissingConfiguration = 52,

    /// <summary>
    /// This error indicates that a device runtime grid launch did not occur because the depth of the child grid would
    /// exceed the maximum supported number of nested grid launches.
    /// </summary>
    CudaErrorLaunchMaxDepthExceeded = 65,

    /// <summary>
    /// This error indicates that a grid launch did not occur because the kernel uses file-scoped textures which are
    /// unsupported by the device runtime. Kernels launched via the device runtime only support textures created with the
    /// Texture Object API's.
    /// </summary>
    CudaErrorLaunchFileScopedTex = 66,

    /// <summary>
    /// This error indicates that a grid launch did not occur because the kernel uses file-scoped surfaces which are
    /// unsupported by the device runtime. Kernels launched via the device runtime only support surfaces created with the
    /// Surface Object API's.
    /// </summary>
    CudaErrorLaunchFileScopedSurf = 67,

    /// <summary>
    /// This error indicates that a call to cudaDeviceSynchronize made from the device runtime failed because the call was
    /// made at grid depth greater than either the default (2 levels of grids) or user specified device limit
    /// cudaLimitDevRuntimeSyncDepth. To be able to synchronize on launched grids at a greater depth successfully, the
    /// maximum nested depth at which cudaDeviceSynchronize will be called must be specified with the
    /// cudaLimitDevRuntimeSyncDepth limit to the cudaDeviceSetLimit api before the host-side launch of a kernel using the
    /// device runtime. Keep in mind that additional levels of sync depth require the runtime to reserve large amounts of
    /// device memory that cannot be used for user allocations.
    /// </summary>
    CudaErrorSyncDepthExceeded = 68,

    /// <summary>
    /// This error indicates that a device runtime grid launch failed because the launch would exceed the limit
    /// cudaLimitDevRuntimePendingLaunchCount. For this launch to proceed successfully, cudaDeviceSetLimit must be called
    /// to set the cudaLimitDevRuntimePendingLaunchCount to be higher than the upper bound of outstanding launches that can
    /// be issued to the device runtime. Keep in mind that raising the limit of pending device runtime launches will
    /// require the runtime to reserve device memory that cannot be used for user allocations.
    /// </summary>
    CudaErrorLaunchPendingCountExceeded = 69,

    /// <summary>
    /// The requested device function does not exist or is not compiled for the proper device architecture.
    /// </summary>
    CudaErrorInvalidDeviceFunction = 98,

    /// <summary>
    /// This indicates that no CUDA-capable devices were detected by the installed CUDA driver.
    /// </summary>
    CudaErrorNoDevice = 100,

    /// <summary>
    /// This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device or that the
    /// action requested is invalid for the specified device.
    /// </summary>
    CudaErrorInvalidDevice = 101,

    /// <summary>
    /// This indicates that the device doesn't have a valid Grid License.
    /// </summary>
    CudaErrorDeviceNotLicensed = 102,

    /// <summary>
    /// By default, the CUDA runtime may perform a minimal set of self-tests, as well as CUDA driver tests, to establish
    /// the validity of both. Introduced in CUDA 11.2, this error return indicates that at least one of these tests has
    /// failed and the validity of either the runtime or the driver could not be established.
    /// </summary>
    CudaErrorSoftwareValidityNotEstablished = 103,

    /// <summary>
    /// This indicates an internal startup failure in the CUDA runtime.
    /// </summary>
    CudaErrorStartupFailure = 127,

    /// <summary>
    /// This indicates that the device kernel image is invalid.
    /// </summary>
    CudaErrorInvalidKernelImage = 200,

    /// <summary>
    /// This most frequently indicates that there is no context bound to the current thread. This can also be returned if
    /// the context passed to an API call is not a valid handle (such as a context that has had cuCtxDestroy() invoked on
    /// it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). See
    /// cuCtxGetApiVersion() for more details.
    /// </summary>
    CudaErrorDeviceUninitialized = 201,

    /// <summary>
    /// This indicates that the buffer object could not be mapped.
    /// </summary>
    CudaErrorMapBufferObjectFailed = 205,

    /// <summary>
    /// This indicates that the buffer object could not be unmapped.
    /// </summary>
    CudaErrorUnmapBufferObjectFailed = 206,

    /// <summary>
    /// This indicates that the specified array is currently mapped and thus cannot be destroyed.
    /// </summary>
    CudaErrorArrayIsMapped = 207,

    /// <summary>
    /// This indicates that the resource is already mapped.
    /// </summary>
    CudaErrorAlreadyMapped = 208,

    /// <summary>
    /// This indicates that there is no kernel image available that is suitable for the device. This can occur when a user
    /// specifies code generation options for a particular CUDA source file that do not include the corresponding device
    /// configuration.
    /// </summary>
    CudaErrorNoKernelImageForDevice = 209,

    /// <summary>
    /// This indicates that a resource has already been acquired.
    /// </summary>
    CudaErrorAlreadyAcquired = 210,

    /// <summary>
    /// This indicates that a resource is not mapped.
    /// </summary>
    CudaErrorNotMapped = 211,

    /// <summary>
    /// This indicates that a mapped resource is not available for access as an array.
    /// </summary>
    CudaErrorNotMappedAsArray = 212,

    /// <summary>
    /// This indicates that a mapped resource is not available for access as a pointer.
    /// </summary>
    CudaErrorNotMappedAsPointer = 213,

    /// <summary>
    /// This indicates that an uncorrectable ECC error was detected during execution.
    /// </summary>
    CudaErrorEccUncorrectable = 214,

    /// <summary>
    /// This indicates that the cudaLimit passed to the API call is not supported by the active device.
    /// </summary>
    CudaErrorUnsupportedLimit = 215,

    /// <summary>
    /// This indicates that a call tried to access an exclusive-thread device that is already in use by a different thread.
    /// </summary>
    CudaErrorDeviceAlreadyInUse = 216,

    /// <summary>
    /// This error indicates that P2P access is not supported across the given devices.
    /// </summary>
    CudaErrorPeerAccessUnsupported = 217,

    /// <summary>
    /// A PTX compilation failed. The runtime may fall back to compiling PTX if an application does not contain a suitable
    /// binary for the current device.
    /// </summary>
    CudaErrorInvalidPtx = 218,

    /// <summary>
    /// This indicates an error with the OpenGL or DirectX context.
    /// </summary>
    CudaErrorInvalidGraphicsContext = 219,

    /// <summary>
    /// This indicates that an uncorrectable NVLink error was detected during the execution.
    /// </summary>
    CudaErrorNvlinkUncorrectable = 220,

    /// <summary>
    /// This indicates that the PTX JIT compiler library was not found. The JIT Compiler library is used for PTX
    /// compilation. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for
    /// the current device.
    /// </summary>
    CudaErrorJitCompilerNotFound = 221,

    /// <summary>
    /// This indicates that the provided PTX was compiled with an unsupported toolchain. The most common reason for this,
    /// is the PTX was generated by a compiler newer than what is supported by the CUDA driver and PTX JIT compiler.
    /// </summary>
    CudaErrorUnsupportedPtxVersion = 222,

    /// <summary>
    /// This indicates that the JIT compilation was disabled. The JIT compilation compiles PTX. The runtime may fall back
    /// to compiling PTX if an application does not contain a suitable binary for the current device.
    /// </summary>
    CudaErrorJitCompilationDisabled = 223,

    /// <summary>
    /// This indicates that the provided execution affinity is not supported by the device.
    /// </summary>
    CudaErrorUnsupportedExecAffinity = 224,

    /// <summary>
    /// This indicates that the device kernel source is invalid.
    /// </summary>
    CudaErrorInvalidSource = 300,

    /// <summary>
    /// This indicates that the file specified was not found.
    /// </summary>
    CudaErrorFileNotFound = 301,

    /// <summary>
    /// This indicates that a link to a shared object failed to resolve.
    /// </summary>
    CudaErrorSharedObjectSymbolNotFound = 302,

    /// <summary>
    /// This indicates that initialization of a shared object failed.
    /// </summary>
    CudaErrorSharedObjectInitFailed = 303,

    /// <summary>
    /// This error indicates that an OS call failed.
    /// </summary>
    CudaErrorOperatingSystem = 304,

    /// <summary>
    /// This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like
    /// cudaStream_t and cudaEvent_t.
    /// </summary>
    CudaErrorInvalidResourceHandle = 400,

    /// <summary>
    /// This indicates that a resource required by the API call is not in a valid state to perform the requested operation.
    /// </summary>
    CudaErrorIllegalState = 401,

    /// <summary>
    /// This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, driver
    /// function names, texture names, and surface names.
    /// </summary>
    CudaErrorSymbolNotFound = 500,

    /// <summary>
    /// This indicates that asynchronous operations issued previously have not completed yet. This result is not actually
    /// an error, but must be indicated differently than cudaSuccess (which indicates completion). Calls that may return
    /// this value include cudaEventQuery() and cudaStreamQuery().
    /// </summary>
    CudaErrorNotReady = 600,

    /// <summary>
    /// The device encountered a load or store instruction on an invalid memory address. This leaves the process in an
    /// inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must
    /// be terminated and relaunched.
    /// </summary>
    CudaErrorIllegalAddress = 700,

    /// <summary>
    /// This indicates that a launch did not occur because it did not have appropriate resources. Although this error is
    /// similar to cudaErrorInvalidConfiguration, this error usually indicates that the user has attempted to pass too many
    /// arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count.
    /// </summary>
    CudaErrorLaunchOutOfResources = 701,

    /// <summary>
    /// This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see
    /// the device property kernelExecTimeoutEnabled for more information. This leaves the process in an inconsistent state
    /// and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and
    /// relaunched.
    /// </summary>
    CudaErrorLaunchTimeout = 702,

    /// <summary>
    /// This error indicates a kernel launch that uses an incompatible texturing mode.
    /// </summary>
    CudaErrorLaunchIncompatibleTexturing = 703,

    /// <summary>
    /// This error indicates that a call to cudaDeviceEnablePeerAccess() is trying to re-enable peer addressing on from a
    /// context which has already had peer addressing enabled.
    /// </summary>
    CudaErrorPeerAccessAlreadyEnabled = 704,

    /// <summary>
    /// This error indicates that cudaDeviceDisablePeerAccess() is trying to disable peer addressing which has not been
    /// enabled yet via cudaDeviceEnablePeerAccess().
    /// </summary>
    CudaErrorPeerAccessNotEnabled = 705,

    /// <summary>
    /// This indicates that the user has called cudaSetValidDevices(), cudaSetDeviceFlags(), cudaD3D9SetDirect3DDevice(),
    /// cudaD3D10SetDirect3DDevice, cudaD3D11SetDirect3DDevice(), or cudaVDPAUSetVDPAUDevice() after initializing the CUDA
    /// runtime by calling non-device management operations (allocating memory and launching kernels are examples of
    /// non-device management operations). This error can also be returned if using runtime/driver interoperability and
    /// there is an existing CUcontext active on the host thread.
    /// </summary>
    CudaErrorSetOnActiveProcess = 708,

    /// <summary>
    /// This error indicates that the context current to the calling thread has been destroyed using cuCtxDestroy, or is a
    /// primary context which has not yet been initialized.
    /// </summary>
    CudaErrorContextIsDestroyed = 709,

    /// <summary>
    /// An assert triggered in device code during kernel execution. The device cannot be used again. All existing
    /// allocations are invalid. To continue using CUDA, the process must be terminated and relaunched.
    /// </summary>
    CudaErrorAssert = 710,

    /// <summary>
    /// This error indicates that the hardware resources required to enable peer access have been exhausted for one or more
    /// of the devices passed to cudaEnablePeerAccess().
    /// </summary>
    CudaErrorTooManyPeers = 711,

    /// <summary>
    /// This error indicates that the memory range passed to cudaHostRegister() has already been registered.
    /// </summary>
    CudaErrorHostMemoryAlreadyRegistered = 712,

    /// <summary>
    /// This error indicates that the pointer passed to cudaHostUnregister() does not correspond to any currently
    /// registered memory region.
    /// </summary>
    CudaErrorHostMemoryNotRegistered = 713,

    /// <summary>
    /// Device encountered an error in the call stack during kernel execution, possibly due to stack corruption or
    /// exceeding the stack count limit. This leaves the process in an inconsistent state and any further CUDA work will
    /// return the same error. To continue using CUDA, the process must be terminated and relaunched.
    /// </summary>
    CudaErrorHardwareStackError = 714,

    /// <summary>
    /// The device encountered an illegal instruction during kernel execution This leaves the process in an inconsistent
    /// state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated
    /// and relaunched.
    /// </summary>
    CudaErrorIllegalInstruction = 715,

    /// <summary>
    /// The device encountered a load or store instruction on a memory address which is not aligned. This leaves the
    /// process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the
    /// process must be terminated and relaunched.
    /// </summary>
    CudaErrorMisalignedAddress = 716,

    /// <summary>
    /// While executing a kernel, the device encountered an instruction which can only operate on memory locations in
    /// certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed
    /// address space. This leaves the process in an inconsistent state and any further CUDA work will return the same
    /// error. To continue using CUDA, the process must be terminated and relaunched.
    /// </summary>
    CudaErrorInvalidAddressSpace = 717,

    /// <summary>
    /// The device encountered an invalid program counter. This leaves the process in an inconsistent state and any further
    /// CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
    /// </summary>
    CudaErrorInvalidPc = 718,

    /// <summary>
    /// An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device
    /// pointer and accessing out of bounds shared memory. Less common cases can be system specific - more information
    /// about these cases can be found in the system specific user guide. This leaves the process in an inconsistent state
    /// and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and
    /// relaunched.
    /// </summary>
    CudaErrorLaunchFailure = 719,

    /// <summary>
    /// This error indicates that the number of blocks launched per grid for a kernel that was launched via either
    /// cudaLaunchCooperativeKernel or cudaLaunchCooperativeKernelMultiDevice exceeds the maximum number of blocks as
    /// allowed by cudaOccupancyMaxActiveBlocksPerMultiprocessor or cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
    /// times the number of multiprocessors as specified by the device attribute cudaDevAttrMultiProcessorCount.
    /// </summary>
    CudaErrorCooperativeLaunchTooLarge = 720,

    /// <summary>
    /// This error indicates the attempted operation is not permitted.
    /// </summary>
    CudaErrorNotPermitted = 800,

    /// <summary>
    /// This error indicates the attempted operation is not supported on the current system or device.
    /// </summary>
    CudaErrorNotSupported = 801,

    /// <summary>
    /// This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the
    /// system configuration is in a valid state and all required driver daemons are actively running. More information
    /// about this error can be found in the system specific user guide.
    /// </summary>
    CudaErrorSystemNotReady = 802,

    /// <summary>
    /// This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer
    /// to the compatibility documentation for supported versions.
    /// </summary>
    CudaErrorSystemDriverMismatch = 803,

    /// <summary>
    /// This error indicates that the system was upgraded to run with forward compatibility but the visible hardware
    /// detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported
    /// hardware matrix or ensure that only supported hardware is visible during initialization via the
    /// CUDA_VISIBLE_DEVICES environment variable.
    /// </summary>
    CudaErrorCompatNotSupportedOnDevice = 804,

    /// <summary>
    /// This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.
    /// </summary>
    CudaErrorMpsConnectionFailed = 805,

    /// <summary>
    /// This error indicates that the remote procedural call between the MPS server and the MPS client failed.
    /// </summary>
    CudaErrorMpsRpcFailure = 806,

    /// <summary>
    /// This error indicates that the MPS server is not ready to accept new MPS client requests. This error can be returned
    /// when the MPS server is in the process of recovering from a fatal failure.
    /// </summary>
    CudaErrorMpsServerNotReady = 807,

    /// <summary>
    /// This error indicates that the hardware resources required to create MPS client have been exhausted.
    /// </summary>
    CudaErrorMpsMaxClientsReached = 808,

    /// <summary>
    /// This error indicates the hardware resources required to device connections have been exhausted.
    /// </summary>
    CudaErrorMpsMaxConnectionsReached = 809,

    /// <summary>
    /// The operation is not permitted when the stream is capturing.
    /// </summary>
    CudaErrorStreamCaptureUnsupported = 900,

    /// <summary>
    /// The current capture sequence on the stream has been invalidated due to a previous error.
    /// </summary>
    CudaErrorStreamCaptureInvalidated = 901,

    /// <summary>
    /// The operation would have resulted in a merge of two independent capture sequences.
    /// </summary>
    CudaErrorStreamCaptureMerge = 902,

    /// <summary>
    /// The capture was not initiated in this stream.
    /// </summary>
    CudaErrorStreamCaptureUnmatched = 903,

    /// <summary>
    /// The capture sequence contains a fork that was not joined to the primary stream.
    /// </summary>
    CudaErrorStreamCaptureUnjoined = 904,

    /// <summary>
    /// A dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering
    /// dependencies are allowed to cross the boundary.
    /// </summary>
    CudaErrorStreamCaptureIsolation = 905,

    /// <summary>
    /// The operation would have resulted in a disallowed implicit dependency on a current capture sequence from
    /// cudaStreamLegacy.
    /// </summary>
    CudaErrorStreamCaptureImplicit = 906,

    /// <summary>
    /// The operation is not permitted on an event which was last recorded in a capturing stream.
    /// </summary>
    CudaErrorCapturedEvent = 907,

    /// <summary>
    /// A stream capture sequence not initiated with the cudaStreamCaptureModeRelaxed argument to cudaStreamBeginCapture
    /// was passed to cudaStreamEndCapture in a different thread.
    /// </summary>
    CudaErrorStreamCaptureWrongThread = 908,

    /// <summary>
    /// This indicates that the wait operation has timed out.
    /// </summary>
    CudaErrorTimeout = 909,

    /// <summary>
    /// This error indicates that the graph update was not performed because it included changes which violated constraints
    /// specific to instantiated graph update.
    /// </summary>
    CudaErrorGraphExecUpdateFailure = 910,

    /// <summary>
    /// This indicates that an async error has occurred in a device outside of CUDA. If CUDA was waiting for an external
    /// device's signal before consuming shared data, the external device signaled an error indicating that the data is not
    /// valid for consumption. This leaves the process in an inconsistent state and any further CUDA work will return the
    /// same error. To continue using CUDA, the process must be terminated and relaunched.
    /// </summary>
    CudaErrorExternalDevice = 911,

    /// <summary>
    /// This indicates that an unknown internal error has occurred.
    /// </summary>
    CudaErrorUnknown = 999
}