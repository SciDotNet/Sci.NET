// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Sci.NET.Accelerators.SourceGenerators.Extensions;

internal static class TypeIdentifierExtensions
{
    public static bool IsSupportedNumberType(this TypeSyntax type)
    {
        if (type is PredefinedTypeSyntax predefinedTypeSyntax)
        {
#pragma warning disable IDE0072
            return predefinedTypeSyntax.Keyword.Kind() switch
#pragma warning restore IDE0072
            {
                SyntaxKind.FloatKeyword => true,
                SyntaxKind.DoubleKeyword => true,
                SyntaxKind.ByteKeyword => true,
                SyntaxKind.SByteKeyword => true,
                SyntaxKind.UShortKeyword => true,
                SyntaxKind.ShortKeyword => true,
                SyntaxKind.UIntKeyword => true,
                SyntaxKind.IntKeyword => true,
                SyntaxKind.ULongKeyword => true,
                SyntaxKind.LongKeyword => true,
                _ => false
            };
        }

        return type is GenericNameSyntax genericNameSyntax && genericNameSyntax.Identifier.ToFullString() != "IMemoryBlock";
    }
}