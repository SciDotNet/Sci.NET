// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using Sci.NET.CUDA.CuBLAS.Exceptions;
using Sci.NET.CUDA.CuBLAS.Types;

namespace Sci.NET.CUDA.CuBLAS.Extensions;

internal static class CublasStatusCodeExceptions
{
    [DebuggerStepThrough]
    public static void Guard(this CublasStatus status)
    {
        switch (status)
        {
            case CublasStatus.CublasSuccess:
                return;
            case CublasStatus.CublasAllocFailed:
                throw new CublasAllocationFailedException();
            case CublasStatus.CublasArchMismatch:
                throw new CublasArchitectureMismatchException();
            case CublasStatus.CublasExecutionFailed:
                throw new CublasExecutionFailedException();
            case CublasStatus.CublasInternalError:
                throw new CublasInternalErrorException();
            case CublasStatus.CublasInvalidValue:
                throw new CublasInvalidValueException();
            case CublasStatus.CublasLicenseError:
                throw new CublasLicenseErrorException();
            case CublasStatus.CublasMappingError:
                throw new CublasMappingErrorException();
            case CublasStatus.CublasNotInitialized:
                throw new CublasNotInitializedException();
            case CublasStatus.CublasNotSupported:
                throw new CublasNotSupportedException();
            default:
                throw new UnreachableException();
        }
    }
}