// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

namespace Sci.NET.Accelerators.SourceGenerators.Extensions;

internal static class SyntaxKindExtensions
{
    public static SyntaxToken ToSyntaxToken(this SyntaxKind kind)
    {
        return SyntaxFactory.Token(kind);
    }
}