// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Sci.NET.Accelerators.SourceGenerator.Translation.Visitors.Expression;

internal class BinaryExpressionSyntaxVisitor : ITranslationVisitor<BinaryExpressionSyntax>
{
    public BinaryExpressionSyntax Visit(BinaryExpressionSyntax syntaxNode, TranslationContext context)
    {
        if (syntaxNode.Left is MemberAccessExpressionSyntax memberAccessExpressionSyntax)
        {
            var left = new MemberAccessExpressionSyntaxVisitor().Visit(memberAccessExpressionSyntax, context);

            return syntaxNode.WithLeft(left);
        }

        if (syntaxNode.Right is MemberAccessExpressionSyntax memberAccessExpressionSyntax2)
        {
            var right = new MemberAccessExpressionSyntaxVisitor().Visit(memberAccessExpressionSyntax2, context);

            return syntaxNode.WithRight(right);
        }

        return syntaxNode;
    }
}