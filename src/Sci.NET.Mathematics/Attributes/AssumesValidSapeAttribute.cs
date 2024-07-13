// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Attributes;

/// <summary>
/// Indicates that the method or property assumes that the <see cref="Shape"/> of the <see cref="ITensor{TNumber}"/> is valid.
/// </summary>
[PublicAPI]
[AttributeUsage(AttributeTargets.Method)]
[ExcludeFromCodeCoverage]
public sealed class AssumesValidShapeAttribute : Attribute
{
}