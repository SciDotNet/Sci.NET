// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.Tensors;

internal static class ServiceProvider
{
    private static ITensorOperationServiceProvider _tensorOperationServiceProvider = new TensorOperationServiceProvider();

    public static ITensorOperationServiceProvider GetTensorOperationServiceProvider()
    {
        return _tensorOperationServiceProvider;
    }

    public static void SetTensorOperationServiceProvider(ITensorOperationServiceProvider tensorOperationServiceProvider)
    {
        _tensorOperationServiceProvider = tensorOperationServiceProvider;
    }
}