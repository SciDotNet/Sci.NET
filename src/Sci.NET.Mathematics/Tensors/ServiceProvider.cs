// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.Tensors;

internal static class ServiceProvider
{
    private static ITensorOperationServiceFactory _tensorOperationServiceFactory = new TensorOperationServiceFactory();

    public static ITensorOperationServiceFactory GetTensorOperationServiceFactory()
    {
        return _tensorOperationServiceFactory;
    }

    public static void SetTensorOperationServiceFactory(ITensorOperationServiceFactory tensorOperationServiceFactory)
    {
        _tensorOperationServiceFactory = tensorOperationServiceFactory;
    }
}