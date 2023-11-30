// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Sci.NET.Accelerators.SourceGenerators.Extensions;

internal static class MemberDeclarationSyntaxExtensions
{
    public static IEnumerable<AttributeSyntax> GetAttributes(this MemberDeclarationSyntax member)
    {
        return from attributeList in member.AttributeLists
            from attribute in attributeList.Attributes
            select attribute;
    }

    public static AttributeSyntax GetFirstAttribute<T>(this MemberDeclarationSyntax member)
        where T : Attribute
    {
        return GetAttributes<T>(member).First();
    }

    public static IEnumerable<AttributeSyntax> GetAttributes<T>(this MemberDeclarationSyntax member)
        where T : Attribute
    {
        return GetAttributes(member).Where(x => x.Name.ToString().EnsureEndsWith("Attribute") == typeof(T).Name);
    }

    public static bool HasAttribute<T>(this MemberDeclarationSyntax member)
        where T : Attribute
    {
        return GetAttributes<T>(member).Any();
    }

    public static bool HasAttribute(this MemberDeclarationSyntax member, string attributeName)
    {
        return GetAttributes(member).Any(x => x.Name.ToString().EnsureEndsWith("Attribute") == attributeName);
    }
}