// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace Sci.NET.Mathematics.Tensors.Exceptions;

/// <summary>
/// The exception that is thrown when a preview feature is not enabled.
/// </summary>
[PublicAPI]
public class PreviewFeatureNotEnabledException : Exception
{
    /// <summary>
    /// Initializes a new instance of the <see cref="PreviewFeatureNotEnabledException"/> class.
    /// </summary>
    /// <param name="featureName">The name of the feature that is not enabled.</param>
    public PreviewFeatureNotEnabledException(string featureName)
        : base($"The feature '{featureName}' is a preview feature and must be enabled to use it. You can enable this feature by calling the corresponding method in the '{nameof(SciDotNetConfiguration.PreviewFeatures)}' class.")
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PreviewFeatureNotEnabledException"/> class.
    /// </summary>
    /// <param name="enabled">Whether the feature is enabled.</param>
    /// <param name="featureName">The name of the feature that is not enabled.</param>
    /// <exception cref="PreviewFeatureNotEnabledException">Thrown when the feature is not enabled.</exception>
    public static void ThrowIfNotEnabled([DoesNotReturnIf(true)] bool enabled, string featureName)
    {
        if (!enabled)
        {
            throw new PreviewFeatureNotEnabledException(featureName);
        }
    }
}