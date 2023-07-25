// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace Sci.NET.Common.Attributes;

/// <summary>
/// An attribute to mark a feature as preview only.
/// </summary>
[AttributeUsage(AttributeTargets.All, Inherited = false, AllowMultiple = true)]
[PublicAPI]
[ExcludeFromCodeCoverage]
public sealed class PreviewFeatureAttribute : Attribute
{
    /// <summary>
    /// Initializes a new instance of the <see cref="PreviewFeatureAttribute"/> class.
    /// </summary>
    /// <param name="reason">The reason the feature is preview only.</param>
    public PreviewFeatureAttribute(string reason = "This feature may not be fully tested.")
    {
        Reason = reason;
    }

    /// <summary>
    /// Gets the reason why this feature is in preview.
    /// </summary>
    public string Reason { get; }
}