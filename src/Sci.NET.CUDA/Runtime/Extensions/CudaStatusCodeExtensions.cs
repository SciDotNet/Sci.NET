// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.CUDA.Runtime.Exceptions;
using Sci.NET.CUDA.Runtime.Structs;

namespace Sci.NET.CUDA.Runtime.Extensions;

internal static class CudaStatusCodeExtensions
{
    public static void EnsureSuccess(this CudaStatusCode status)
    {
        if (status != CudaStatusCode.CudaSuccess)
        {
            throw new CudaRuntimeException(status);
        }
    }
}