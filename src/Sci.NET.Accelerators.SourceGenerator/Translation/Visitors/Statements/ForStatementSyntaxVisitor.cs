// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Sci.NET.Accelerators.SourceGenerator.Translation.Visitors.Expression;

namespace Sci.NET.Accelerators.SourceGenerator.Translation.Visitors.Statements;

internal class ForStatementSyntaxVisitor : ITranslationVisitor<ForStatementSyntax>
{
    public ForStatementSyntax Visit(ForStatementSyntax syntaxNode, TranslationContext context)
    {
        var statements = new List<StatementSyntax>();

        var condition = syntaxNode.Condition switch
        {
            BinaryExpressionSyntax binaryExpressionSyntax => new BinaryExpressionSyntaxVisitor().Visit(binaryExpressionSyntax, context),
            _ => syntaxNode.Condition
        };

        switch (syntaxNode.Statement)
        {
            case BlockSyntax blockSyntax:
                statements.AddRange(blockSyntax.Statements.Select(statement => new StatementVisitor().Visit(statement, context)));
                break;
            default:
                statements.Add(syntaxNode.Statement);
                break;
        }

        statements.Insert(
            0,
            SyntaxFactory.ExpressionStatement(
                SyntaxFactory.InvocationExpression(
                    SyntaxFactory.MemberAccessExpression(
                        SyntaxKind.SimpleMemberAccessExpression,
                        SyntaxFactory.IdentifierName("Sci"),
                        SyntaxFactory.IdentifierName("BeginLoop")),
                    SyntaxFactory.ArgumentList(
                        SyntaxFactory.SeparatedList(
                            new ArgumentSyntax[]
                            {
                                SyntaxFactory.Argument(
                                    syntaxNode.Condition ??
                                    SyntaxFactory.BaseExpression(SyntaxFactory.Token(SyntaxKind.NullKeyword)))
                            })))));

        statements.Add(
            SyntaxFactory.ExpressionStatement(
                SyntaxFactory.InvocationExpression(
                    SyntaxFactory.MemberAccessExpression(
                        SyntaxKind.SimpleMemberAccessExpression,
                        SyntaxFactory.IdentifierName("Sci"),
                        SyntaxFactory.IdentifierName("EndLoop")))));

        return syntaxNode
            .WithCondition(condition)
            .WithStatement(SyntaxFactory.Block(statements));
    }
}