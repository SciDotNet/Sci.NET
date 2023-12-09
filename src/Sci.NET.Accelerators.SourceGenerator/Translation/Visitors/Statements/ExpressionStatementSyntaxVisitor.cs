// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis.CSharp.Syntax;
using Sci.NET.Accelerators.SourceGenerator.Translation.Visitors.Expression;

namespace Sci.NET.Accelerators.SourceGenerator.Translation.Visitors.Statements;

internal class ExpressionStatementSyntaxVisitor : ITranslationVisitor<ExpressionStatementSyntax>
{
    public ExpressionStatementSyntax Visit(ExpressionStatementSyntax syntaxNode, TranslationContext context)
    {
        if (syntaxNode.Expression is not InvocationExpressionSyntax invocationExpressionSyntax)
        {
            return syntaxNode;
        }

        if (syntaxNode.Parent is not IdentifierNameSyntax)
        {
            return syntaxNode;
        }

        var expression = invocationExpressionSyntax.Expression switch
        {
            MemberAccessExpressionSyntax memberAccessExpressionSyntax => new MemberAccessExpressionSyntaxVisitor().Visit(memberAccessExpressionSyntax, context),
            _ => syntaxNode.Expression
        };

        return syntaxNode.WithExpression(expression);
    }
}