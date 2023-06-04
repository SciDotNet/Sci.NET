// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.CUDA.Native;

namespace Sci.NET.CUDA.Runtime;

internal static class RuntimeApiResolver
{
    private static readonly ICudaRuntimeApi CudaRuntimeApi = new CudaRuntimeApiV12();

    public static ICudaRuntimeApi Resolve()
    {
        return CudaRuntimeApi;
    }
}