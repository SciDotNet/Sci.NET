// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// A service provider for tensor operations.
/// </summary>
[PublicAPI]
public static class TensorServiceProvider
{
    private static ITensorOperationServiceProvider _tensorOperationServiceProvider = new TensorOperationServiceProvider();

    /// <summary>
    /// Gets the current tensor operation service provider.
    /// </summary>
    /// <returns>The <see cref="ITensorOperationServiceProvider"/>.</returns>
    [DebuggerStepThrough]
#pragma warning disable CA1024
    public static ITensorOperationServiceProvider GetTensorOperationServiceProvider()
#pragma warning restore CA1024
    {
        return _tensorOperationServiceProvider;
    }

    [DebuggerStepThrough]
    internal static void SetTensorOperationServiceProvider(ITensorOperationServiceProvider tensorOperationServiceProvider)
    {
        _tensorOperationServiceProvider = tensorOperationServiceProvider;
    }
}