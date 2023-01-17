// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.CUDA.RuntimeApi.Bindings.Exceptions;

/// <summary>
/// This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.
/// </summary>
[PublicAPI]
public class CudaProfilerDisabledException : CudaException
{
    private const string DefaultMessage =
        "This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.";

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaProfilerDisabledException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.
    /// </remarks>
    public CudaProfilerDisabledException()
        : base(DefaultMessage)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CudaProfilerDisabledException"/> class.
    /// </summary>
    /// <remarks>
    /// This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.
    /// </remarks>
    /// <param name="innerException">The inner exception.</param>
    public CudaProfilerDisabledException(Exception innerException)
        : base(DefaultMessage, innerException)
    {
    }
}
