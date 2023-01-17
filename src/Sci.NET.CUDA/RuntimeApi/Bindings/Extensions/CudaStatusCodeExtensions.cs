// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;
using Sci.NET.CUDA.RuntimeApi.Bindings.Types;

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Extensions;

#pragma warning disable CA1506 // Avoid excessive class coupling
internal static class CudaStatusCodeExtensions
{
    public static bool IsFailureCode(this CudaStatusCode code)
    {
        return code != CudaStatusCode.CudaSuccess;
    }

    [ExcludeFromCodeCoverage]
    public static Exception GetException(this CudaStatusCode runtimeStatusCodeCode)
    {
#pragma warning disable IDE0072
        return runtimeStatusCodeCode switch
#pragma warning restore IDE0072
        {
            CudaStatusCode.CudaErrorInvalidValue => new CudaInvalidValueException(),
            CudaStatusCode.CudaErrorMemoryAllocation => new CudaMemoryAllocationException(),
            CudaStatusCode.CudaErrorInitializationError => new CudaInitializationErrorException(),
            CudaStatusCode.CudaErrorCudartUnloading => new CudaCudartUnloadingException(),
            CudaStatusCode.CudaErrorProfilerDisabled => new CudaProfilerDisabledException(),
            CudaStatusCode.CudaErrorInvalidConfiguration => new CudaInvalidConfigurationException(),
            CudaStatusCode.CudaErrorInvalidPitchValue => new CudaInvalidPitchValueException(),
            CudaStatusCode.CudaErrorInvalidSymbol => new CudaInvalidSymbolException(),
            CudaStatusCode.CudaErrorInvalidDevicePointer => new CudaInvalidDevicePointerException(),
            CudaStatusCode.CudaErrorInvalidTexture => new CudaInvalidTextureException(),
            CudaStatusCode.CudaErrorInvalidTextureBinding => new CudaInvalidTextureBindingException(),
            CudaStatusCode.CudaErrorInvalidChannelDescriptor => new CudaInvalidChannelDescriptorException(),
            CudaStatusCode.CudaErrorInvalidMemcpyDirection => new CudaInvalidMemcpyDirectionException(),
            CudaStatusCode.CudaErrorInvalidFilterSetting => new CudaInvalidFilterSettingException(),
            CudaStatusCode.CudaErrorInvalidNormSetting => new CudaInvalidNormSettingException(),
            CudaStatusCode.CudaErrorStubLibrary => new CudaStubLibraryException(),
            CudaStatusCode.CudaErrorInsufficientDriver => new CudaInsufficientDriverException(),
            CudaStatusCode.CudaErrorCallRequiresNewerDriver => new CudaCallRequiresNewerDriverException(),
            CudaStatusCode.CudaErrorInvalidSurface => new CudaInvalidSurfaceException(),
            CudaStatusCode.CudaErrorDuplicateVariableName => new CudaDuplicateVariableNameException(),
            CudaStatusCode.CudaErrorDuplicateTextureName => new CudaDuplicateTextureNameException(),
            CudaStatusCode.CudaErrorDuplicateSurfaceName => new CudaDuplicateSurfaceNameException(),
            CudaStatusCode.CudaErrorDevicesUnavailable => new CudaDevicesUnavailableException(),
            CudaStatusCode.CudaErrorIncompatibleDriverContext => new CudaIncompatibleDriverContextException(),
            CudaStatusCode.CudaErrorMissingConfiguration => new CudaMissingConfigurationException(),
            CudaStatusCode.CudaErrorLaunchMaxDepthExceeded => new CudaLaunchMaxDepthExceededException(),
            CudaStatusCode.CudaErrorLaunchFileScopedTex => new CudaLaunchFileScopedTexException(),
            CudaStatusCode.CudaErrorLaunchFileScopedSurf => new CudaLaunchFileScopedSurfException(),
            CudaStatusCode.CudaErrorSyncDepthExceeded => new CudaSyncDepthExceededException(),
            CudaStatusCode.CudaErrorLaunchPendingCountExceeded => new CudaLaunchPendingCountExceededException(),
            CudaStatusCode.CudaErrorInvalidDeviceFunction => new CudaInvalidDeviceFunctionException(),
            CudaStatusCode.CudaErrorNoDevice => new CudaNoDeviceException(),
            CudaStatusCode.CudaErrorInvalidDevice => new CudaInvalidDeviceException(),
            CudaStatusCode.CudaErrorDeviceNotLicensed => new CudaDeviceNotLicensedException(),
            CudaStatusCode.CudaErrorSoftwareValidityNotEstablished => new CudaSoftwareValidityNotEstablishedException(),
            CudaStatusCode.CudaErrorStartupFailure => new CudaStartupFailureException(),
            CudaStatusCode.CudaErrorInvalidKernelImage => new CudaInvalidKernelImageException(),
            CudaStatusCode.CudaErrorDeviceUninitialized => new CudaDeviceUninitializedException(),
            CudaStatusCode.CudaErrorMapBufferObjectFailed => new CudaMapBufferObjectFailedException(),
            CudaStatusCode.CudaErrorUnmapBufferObjectFailed => new CudaUnmapBufferObjectFailedException(),
            CudaStatusCode.CudaErrorArrayIsMapped => new CudaArrayIsMappedException(),
            CudaStatusCode.CudaErrorAlreadyMapped => new CudaAlreadyMappedException(),
            CudaStatusCode.CudaErrorNoKernelImageForDevice => new CudaNoKernelImageForDeviceException(),
            CudaStatusCode.CudaErrorAlreadyAcquired => new CudaAlreadyAcquiredException(),
            CudaStatusCode.CudaErrorNotMapped => new CudaNotMappedException(),
            CudaStatusCode.CudaErrorNotMappedAsArray => new CudaNotMappedAsArrayException(),
            CudaStatusCode.CudaErrorNotMappedAsPointer => new CudaNotMappedAsPointerException(),
            CudaStatusCode.CudaErrorEccUncorrectable => new CudaEccUncorrectableException(),
            CudaStatusCode.CudaErrorUnsupportedLimit => new CudaUnsupportedLimitException(),
            CudaStatusCode.CudaErrorDeviceAlreadyInUse => new CudaDeviceAlreadyInUseException(),
            CudaStatusCode.CudaErrorPeerAccessUnsupported => new CudaPeerAccessUnsupportedException(),
            CudaStatusCode.CudaErrorInvalidPtx => new CudaInvalidPtxException(),
            CudaStatusCode.CudaErrorInvalidGraphicsContext => new CudaInvalidGraphicsContextException(),
            CudaStatusCode.CudaErrorNvlinkUncorrectable => new CudaNvlinkUncorrectableException(),
            CudaStatusCode.CudaErrorJitCompilerNotFound => new CudaJitCompilerNotFoundException(),
            CudaStatusCode.CudaErrorUnsupportedPtxVersion => new CudaUnsupportedPtxVersionException(),
            CudaStatusCode.CudaErrorJitCompilationDisabled => new CudaJitCompilationDisabledException(),
            CudaStatusCode.CudaErrorUnsupportedExecAffinity => new CudaUnsupportedExecAffinityException(),
            CudaStatusCode.CudaErrorInvalidSource => new CudaInvalidSourceException(),
            CudaStatusCode.CudaErrorFileNotFound => new CudaFileNotFoundException(),
            CudaStatusCode.CudaErrorSharedObjectSymbolNotFound => new CudaSharedObjectSymbolNotFoundException(),
            CudaStatusCode.CudaErrorSharedObjectInitFailed => new CudaSharedObjectInitFailedException(),
            CudaStatusCode.CudaErrorOperatingSystem => new CudaOperatingSystemException(),
            CudaStatusCode.CudaErrorInvalidResourceHandle => new CudaInvalidResourceHandleException(),
            CudaStatusCode.CudaErrorIllegalState => new CudaIllegalStateException(),
            CudaStatusCode.CudaErrorSymbolNotFound => new CudaSymbolNotFoundException(),
            CudaStatusCode.CudaErrorNotReady => new CudaNotReadyException(),
            CudaStatusCode.CudaErrorIllegalAddress => new CudaIllegalAddressException(),
            CudaStatusCode.CudaErrorLaunchOutOfResources => new CudaLaunchOutOfResourcesException(),
            CudaStatusCode.CudaErrorLaunchTimeout => new CudaLaunchTimeoutException(),
            CudaStatusCode.CudaErrorLaunchIncompatibleTexturing => new CudaLaunchIncompatibleTexturingException(),
            CudaStatusCode.CudaErrorPeerAccessAlreadyEnabled => new CudaPeerAccessAlreadyEnabledException(),
            CudaStatusCode.CudaErrorPeerAccessNotEnabled => new CudaPeerAccessNotEnabledException(),
            CudaStatusCode.CudaErrorSetOnActiveProcess => new CudaSetOnActiveProcessException(),
            CudaStatusCode.CudaErrorContextIsDestroyed => new CudaContextIsDestroyedException(),
            CudaStatusCode.CudaErrorAssert => new CudaAssertException(),
            CudaStatusCode.CudaErrorTooManyPeers => new CudaTooManyPeersException(),
            CudaStatusCode.CudaErrorHostMemoryAlreadyRegistered => new CudaHostMemoryAlreadyRegisteredException(),
            CudaStatusCode.CudaErrorHostMemoryNotRegistered => new CudaHostMemoryNotRegisteredException(),
            CudaStatusCode.CudaErrorHardwareStackError => new CudaHardwareStackErrorException(),
            CudaStatusCode.CudaErrorIllegalInstruction => new CudaIllegalInstructionException(),
            CudaStatusCode.CudaErrorMisalignedAddress => new CudaMisalignedAddressException(),
            CudaStatusCode.CudaErrorInvalidAddressSpace => new CudaInvalidAddressSpaceException(),
            CudaStatusCode.CudaErrorInvalidPc => new CudaInvalidPcException(),
            CudaStatusCode.CudaErrorLaunchFailure => new CudaLaunchFailureException(),
            CudaStatusCode.CudaErrorCooperativeLaunchTooLarge => new CudaCooperativeLaunchTooLargeException(),
            CudaStatusCode.CudaErrorNotPermitted => new CudaNotPermittedException(),
            CudaStatusCode.CudaErrorNotSupported => new CudaNotSupportedException(),
            CudaStatusCode.CudaErrorSystemNotReady => new CudaSystemNotReadyException(),
            CudaStatusCode.CudaErrorSystemDriverMismatch => new CudaSystemDriverMismatchException(),
            CudaStatusCode.CudaErrorCompatNotSupportedOnDevice => new CudaCompatNotSupportedOnDeviceException(),
            CudaStatusCode.CudaErrorMpsConnectionFailed => new CudaMpsConnectionFailedException(),
            CudaStatusCode.CudaErrorMpsRpcFailure => new CudaMpsRpcFailureException(),
            CudaStatusCode.CudaErrorMpsServerNotReady => new CudaMpsServerNotReadyException(),
            CudaStatusCode.CudaErrorMpsMaxClientsReached => new CudaMpsMaxClientsReachedException(),
            CudaStatusCode.CudaErrorMpsMaxConnectionsReached => new CudaMpsMaxConnectionsReachedException(),
            CudaStatusCode.CudaErrorStreamCaptureUnsupported => new CudaStreamCaptureUnsupportedException(),
            CudaStatusCode.CudaErrorStreamCaptureInvalidated => new CudaStreamCaptureInvalidatedException(),
            CudaStatusCode.CudaErrorStreamCaptureMerge => new CudaStreamCaptureMergeException(),
            CudaStatusCode.CudaErrorStreamCaptureUnmatched => new CudaStreamCaptureUnmatchedException(),
            CudaStatusCode.CudaErrorStreamCaptureUnjoined => new CudaStreamCaptureUnjoinedException(),
            CudaStatusCode.CudaErrorStreamCaptureIsolation => new CudaStreamCaptureIsolationException(),
            CudaStatusCode.CudaErrorStreamCaptureImplicit => new CudaStreamCaptureImplicitException(),
            CudaStatusCode.CudaErrorCapturedEvent => new CudaCapturedEventException(),
            CudaStatusCode.CudaErrorStreamCaptureWrongThread => new CudaStreamCaptureWrongThreadException(),
            CudaStatusCode.CudaErrorTimeout => new CudaTimeoutException(),
            CudaStatusCode.CudaErrorGraphExecUpdateFailure => new CudaGraphExecUpdateFailureException(),
            CudaStatusCode.CudaErrorExternalDevice => new CudaExternalDeviceException(),
            CudaStatusCode.CudaErrorUnknown => new CudaUnknownException(),
            _ => new CudaUnknownException(),
        };
    }

    [PublicAPI]
    [ExcludeFromCodeCoverage]
    public static Exception GetException(this CudaStatusCode runtimeStatusCodeCode, Exception innerException)
    {
#pragma warning disable IDE0072
        return runtimeStatusCodeCode switch
#pragma warning restore IDE0072
        {
            CudaStatusCode.CudaErrorInvalidValue => new CudaInvalidValueException(innerException),
            CudaStatusCode.CudaErrorMemoryAllocation => new CudaMemoryAllocationException(innerException),
            CudaStatusCode.CudaErrorInitializationError => new CudaInitializationErrorException(innerException),
            CudaStatusCode.CudaErrorCudartUnloading => new CudaCudartUnloadingException(innerException),
            CudaStatusCode.CudaErrorProfilerDisabled => new CudaProfilerDisabledException(innerException),
            CudaStatusCode.CudaErrorInvalidConfiguration => new CudaInvalidConfigurationException(innerException),
            CudaStatusCode.CudaErrorInvalidPitchValue => new CudaInvalidPitchValueException(innerException),
            CudaStatusCode.CudaErrorInvalidSymbol => new CudaInvalidSymbolException(innerException),
            CudaStatusCode.CudaErrorInvalidDevicePointer => new CudaInvalidDevicePointerException(innerException),
            CudaStatusCode.CudaErrorInvalidTexture => new CudaInvalidTextureException(innerException),
            CudaStatusCode.CudaErrorInvalidTextureBinding => new CudaInvalidTextureBindingException(innerException),
            CudaStatusCode.CudaErrorInvalidChannelDescriptor => new CudaInvalidChannelDescriptorException(
                innerException),
            CudaStatusCode.CudaErrorInvalidMemcpyDirection => new CudaInvalidMemcpyDirectionException(innerException),
            CudaStatusCode.CudaErrorInvalidFilterSetting => new CudaInvalidFilterSettingException(innerException),
            CudaStatusCode.CudaErrorInvalidNormSetting => new CudaInvalidNormSettingException(innerException),
            CudaStatusCode.CudaErrorStubLibrary => new CudaStubLibraryException(innerException),
            CudaStatusCode.CudaErrorInsufficientDriver => new CudaInsufficientDriverException(innerException),
            CudaStatusCode.CudaErrorCallRequiresNewerDriver => new CudaCallRequiresNewerDriverException(innerException),
            CudaStatusCode.CudaErrorInvalidSurface => new CudaInvalidSurfaceException(innerException),
            CudaStatusCode.CudaErrorDuplicateVariableName => new CudaDuplicateVariableNameException(innerException),
            CudaStatusCode.CudaErrorDuplicateTextureName => new CudaDuplicateTextureNameException(innerException),
            CudaStatusCode.CudaErrorDuplicateSurfaceName => new CudaDuplicateSurfaceNameException(innerException),
            CudaStatusCode.CudaErrorDevicesUnavailable => new CudaDevicesUnavailableException(innerException),
            CudaStatusCode.CudaErrorIncompatibleDriverContext => new CudaIncompatibleDriverContextException(
                innerException),
            CudaStatusCode.CudaErrorMissingConfiguration => new CudaMissingConfigurationException(innerException),
            CudaStatusCode.CudaErrorLaunchMaxDepthExceeded => new CudaLaunchMaxDepthExceededException(innerException),
            CudaStatusCode.CudaErrorLaunchFileScopedTex => new CudaLaunchFileScopedTexException(innerException),
            CudaStatusCode.CudaErrorLaunchFileScopedSurf => new CudaLaunchFileScopedSurfException(innerException),
            CudaStatusCode.CudaErrorSyncDepthExceeded => new CudaSyncDepthExceededException(innerException),
            CudaStatusCode.CudaErrorLaunchPendingCountExceeded => new CudaLaunchPendingCountExceededException(
                innerException),
            CudaStatusCode.CudaErrorInvalidDeviceFunction => new CudaInvalidDeviceFunctionException(innerException),
            CudaStatusCode.CudaErrorNoDevice => new CudaNoDeviceException(innerException),
            CudaStatusCode.CudaErrorInvalidDevice => new CudaInvalidDeviceException(innerException),
            CudaStatusCode.CudaErrorDeviceNotLicensed => new CudaDeviceNotLicensedException(innerException),
            CudaStatusCode.CudaErrorSoftwareValidityNotEstablished => new CudaSoftwareValidityNotEstablishedException(
                innerException),
            CudaStatusCode.CudaErrorStartupFailure => new CudaStartupFailureException(innerException),
            CudaStatusCode.CudaErrorInvalidKernelImage => new CudaInvalidKernelImageException(innerException),
            CudaStatusCode.CudaErrorDeviceUninitialized => new CudaDeviceUninitializedException(innerException),
            CudaStatusCode.CudaErrorMapBufferObjectFailed => new CudaMapBufferObjectFailedException(innerException),
            CudaStatusCode.CudaErrorUnmapBufferObjectFailed => new CudaUnmapBufferObjectFailedException(innerException),
            CudaStatusCode.CudaErrorArrayIsMapped => new CudaArrayIsMappedException(innerException),
            CudaStatusCode.CudaErrorAlreadyMapped => new CudaAlreadyMappedException(innerException),
            CudaStatusCode.CudaErrorNoKernelImageForDevice => new CudaNoKernelImageForDeviceException(innerException),
            CudaStatusCode.CudaErrorAlreadyAcquired => new CudaAlreadyAcquiredException(innerException),
            CudaStatusCode.CudaErrorNotMapped => new CudaNotMappedException(innerException),
            CudaStatusCode.CudaErrorNotMappedAsArray => new CudaNotMappedAsArrayException(innerException),
            CudaStatusCode.CudaErrorNotMappedAsPointer => new CudaNotMappedAsPointerException(innerException),
            CudaStatusCode.CudaErrorEccUncorrectable => new CudaEccUncorrectableException(innerException),
            CudaStatusCode.CudaErrorUnsupportedLimit => new CudaUnsupportedLimitException(innerException),
            CudaStatusCode.CudaErrorDeviceAlreadyInUse => new CudaDeviceAlreadyInUseException(innerException),
            CudaStatusCode.CudaErrorPeerAccessUnsupported => new CudaPeerAccessUnsupportedException(innerException),
            CudaStatusCode.CudaErrorInvalidPtx => new CudaInvalidPtxException(innerException),
            CudaStatusCode.CudaErrorInvalidGraphicsContext => new CudaInvalidGraphicsContextException(innerException),
            CudaStatusCode.CudaErrorNvlinkUncorrectable => new CudaNvlinkUncorrectableException(innerException),
            CudaStatusCode.CudaErrorJitCompilerNotFound => new CudaJitCompilerNotFoundException(innerException),
            CudaStatusCode.CudaErrorUnsupportedPtxVersion => new CudaUnsupportedPtxVersionException(innerException),
            CudaStatusCode.CudaErrorJitCompilationDisabled => new CudaJitCompilationDisabledException(innerException),
            CudaStatusCode.CudaErrorUnsupportedExecAffinity => new CudaUnsupportedExecAffinityException(innerException),
            CudaStatusCode.CudaErrorInvalidSource => new CudaInvalidSourceException(innerException),
            CudaStatusCode.CudaErrorFileNotFound => new CudaFileNotFoundException(innerException),
            CudaStatusCode.CudaErrorSharedObjectSymbolNotFound => new CudaSharedObjectSymbolNotFoundException(
                innerException),
            CudaStatusCode.CudaErrorSharedObjectInitFailed => new CudaSharedObjectInitFailedException(innerException),
            CudaStatusCode.CudaErrorOperatingSystem => new CudaOperatingSystemException(innerException),
            CudaStatusCode.CudaErrorInvalidResourceHandle => new CudaInvalidResourceHandleException(innerException),
            CudaStatusCode.CudaErrorIllegalState => new CudaIllegalStateException(innerException),
            CudaStatusCode.CudaErrorSymbolNotFound => new CudaSymbolNotFoundException(innerException),
            CudaStatusCode.CudaErrorNotReady => new CudaNotReadyException(innerException),
            CudaStatusCode.CudaErrorIllegalAddress => new CudaIllegalAddressException(innerException),
            CudaStatusCode.CudaErrorLaunchOutOfResources => new CudaLaunchOutOfResourcesException(innerException),
            CudaStatusCode.CudaErrorLaunchTimeout => new CudaLaunchTimeoutException(innerException),
            CudaStatusCode.CudaErrorLaunchIncompatibleTexturing => new CudaLaunchIncompatibleTexturingException(
                innerException),
            CudaStatusCode.CudaErrorPeerAccessAlreadyEnabled => new CudaPeerAccessAlreadyEnabledException(
                innerException),
            CudaStatusCode.CudaErrorPeerAccessNotEnabled => new CudaPeerAccessNotEnabledException(innerException),
            CudaStatusCode.CudaErrorSetOnActiveProcess => new CudaSetOnActiveProcessException(innerException),
            CudaStatusCode.CudaErrorContextIsDestroyed => new CudaContextIsDestroyedException(innerException),
            CudaStatusCode.CudaErrorAssert => new CudaAssertException(innerException),
            CudaStatusCode.CudaErrorTooManyPeers => new CudaTooManyPeersException(innerException),
            CudaStatusCode.CudaErrorHostMemoryAlreadyRegistered => new CudaHostMemoryAlreadyRegisteredException(
                innerException),
            CudaStatusCode.CudaErrorHostMemoryNotRegistered => new CudaHostMemoryNotRegisteredException(innerException),
            CudaStatusCode.CudaErrorHardwareStackError => new CudaHardwareStackErrorException(innerException),
            CudaStatusCode.CudaErrorIllegalInstruction => new CudaIllegalInstructionException(innerException),
            CudaStatusCode.CudaErrorMisalignedAddress => new CudaMisalignedAddressException(innerException),
            CudaStatusCode.CudaErrorInvalidAddressSpace => new CudaInvalidAddressSpaceException(innerException),
            CudaStatusCode.CudaErrorInvalidPc => new CudaInvalidPcException(innerException),
            CudaStatusCode.CudaErrorLaunchFailure => new CudaLaunchFailureException(innerException),
            CudaStatusCode.CudaErrorCooperativeLaunchTooLarge => new CudaCooperativeLaunchTooLargeException(
                innerException),
            CudaStatusCode.CudaErrorNotPermitted => new CudaNotPermittedException(innerException),
            CudaStatusCode.CudaErrorNotSupported => new CudaNotSupportedException(innerException),
            CudaStatusCode.CudaErrorSystemNotReady => new CudaSystemNotReadyException(innerException),
            CudaStatusCode.CudaErrorSystemDriverMismatch => new CudaSystemDriverMismatchException(innerException),
            CudaStatusCode.CudaErrorCompatNotSupportedOnDevice => new CudaCompatNotSupportedOnDeviceException(
                innerException),
            CudaStatusCode.CudaErrorMpsConnectionFailed => new CudaMpsConnectionFailedException(innerException),
            CudaStatusCode.CudaErrorMpsRpcFailure => new CudaMpsRpcFailureException(innerException),
            CudaStatusCode.CudaErrorMpsServerNotReady => new CudaMpsServerNotReadyException(innerException),
            CudaStatusCode.CudaErrorMpsMaxClientsReached => new CudaMpsMaxClientsReachedException(innerException),
            CudaStatusCode.CudaErrorMpsMaxConnectionsReached => new CudaMpsMaxConnectionsReachedException(
                innerException),
            CudaStatusCode.CudaErrorStreamCaptureUnsupported => new CudaStreamCaptureUnsupportedException(
                innerException),
            CudaStatusCode.CudaErrorStreamCaptureInvalidated => new CudaStreamCaptureInvalidatedException(
                innerException),
            CudaStatusCode.CudaErrorStreamCaptureMerge => new CudaStreamCaptureMergeException(innerException),
            CudaStatusCode.CudaErrorStreamCaptureUnmatched => new CudaStreamCaptureUnmatchedException(innerException),
            CudaStatusCode.CudaErrorStreamCaptureUnjoined => new CudaStreamCaptureUnjoinedException(innerException),
            CudaStatusCode.CudaErrorStreamCaptureIsolation => new CudaStreamCaptureIsolationException(innerException),
            CudaStatusCode.CudaErrorStreamCaptureImplicit => new CudaStreamCaptureImplicitException(innerException),
            CudaStatusCode.CudaErrorCapturedEvent => new CudaCapturedEventException(innerException),
            CudaStatusCode.CudaErrorStreamCaptureWrongThread => new CudaStreamCaptureWrongThreadException(
                innerException),
            CudaStatusCode.CudaErrorTimeout => new CudaTimeoutException(innerException),
            CudaStatusCode.CudaErrorGraphExecUpdateFailure => new CudaGraphExecUpdateFailureException(innerException),
            CudaStatusCode.CudaErrorExternalDevice => new CudaExternalDeviceException(innerException),
            CudaStatusCode.CudaErrorUnknown => new CudaUnknownException(innerException),
            _ => new CudaUnknownException(innerException),
        };
    }

    public static void Guard(this CudaStatusCode runtimeStatusCode)
    {
        if (runtimeStatusCode.IsFailureCode())
        {
            throw runtimeStatusCode.GetException();
        }
    }
}

#pragma warning restore CA1506