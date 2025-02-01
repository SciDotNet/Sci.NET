// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace Sci.NET.Mathematics;

/// <summary>
/// Configuration for Sci.NET.
/// </summary>
[PublicAPI]
[SuppressMessage("Design", "CA1034:Nested types should not be visible", Justification = "Configuration class")]
public static class SciDotNetConfiguration
{
    /// <summary>
    /// Configuration for preview features.
    /// </summary>
    public static class PreviewFeatures
    {
        /// <summary>
        /// Gets a value indicating whether the auto-grad feature is enabled.
        /// </summary>
        public static bool AutoGradEnabled { get; private set; }

        /// <summary>
        /// Enables the auto-grad feature.
        /// </summary>
        public static void EnableAutoGrad()
        {
            AutoGradEnabled = true;
        }
    }
}