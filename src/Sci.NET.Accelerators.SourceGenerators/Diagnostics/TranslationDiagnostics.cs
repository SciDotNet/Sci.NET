// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Sci.NET.Accelerators.SourceGenerators.Diagnostics;

internal static class TranslationDiagnostics
{
    private static readonly DiagnosticDescriptor ReturnTypeNotSupportedDiagnostic = new (
        "SCI0001",
        "The return type is not supported for kernel methods",
        "The return type {0} is not supported for kernel methods",
        "Sci.NET.Translation",
        DiagnosticSeverity.Error,
        true,
        description: "Only void return types are supported for kernel methods.",
        helpLinkUri: "https://docs.scidotnet.org/analysers/SCI0001");

    private static readonly DiagnosticDescriptor KernelExpressionBodyNotSupported = new (
        "SCI0002",
        "Kernel methods cannot have expression bodies",
        "Kernel methods cannot have expression bodies",
        "Sci.NET.Translation",
        DiagnosticSeverity.Error,
        true,
        description: "Kernel methods cannot have expression bodies.",
        helpLinkUri: "https://docs.scidotnet.org/analysers/SCI0002");

    private static readonly DiagnosticDescriptor EmptyKernelBody = new (
        "SCI0003",
        "Kernel methods cannot have empty bodies",
        "Kernel methods cannot have empty bodies",
        "Sci.NET.Translation",
        DiagnosticSeverity.Error,
        true,
        description: "Kernel methods cannot have empty bodies.",
        helpLinkUri: "https://docs.scidotnet.org/analysers/SCI0003");

    private static readonly DiagnosticDescriptor KernelParameterModifiersNotSupported = new (
        "SCI0004",
        "Parameter modifier not supported for kernel methods",
        "Kernel method parameter '{0}' is not supported",
        "Sci.NET.Translation",
        DiagnosticSeverity.Error,
        true,
        description: "Kernel method parameters cannot have modifiers.",
        helpLinkUri: "https://docs.scidotnet.org/analysers/SCI0004");

    private static readonly DiagnosticDescriptor ParameterTypeNotSupported = new (
        "SCI0005",
        "Parameter type not supported for kernel methods",
        "Kernel method parameter '{0}' is not supported",
        "Sci.NET.Translation",
        DiagnosticSeverity.Error,
        true,
        description: "Kernel method parameters cannot have modifiers.",
        helpLinkUri: "https://docs.scidotnet.org/analysers/SCI0005");

    private static readonly DiagnosticDescriptor StatementTypeNotSupported = new (
        "SCI0006",
        "Statement type not supported for kernel methods",
        "Kernel method statement '{0}' is not supported",
        "Sci.NET.Translation",
        DiagnosticSeverity.Error,
        true,
        description: "Kernel method statements cannot have modifiers.",
        helpLinkUri: "https://docs.scidotnet.org/analysers/SCI0006");

    private static readonly DiagnosticDescriptor InvalidAssignmentExpression = new (
        "SCI0007",
        "Invalid assignment expression",
        "Invalid assignment expression '{0}'",
        "Sci.NET.Translation",
        DiagnosticSeverity.Error,
        true,
        description: "Invalid assignment expression.",
        helpLinkUri: "https://docs.scidotnet.org/analysers/SCI0007");

    public static void ReportKernelExpressionBodyNotSupported(ArrowExpressionClauseSyntax expression, SourceProductionContext context)
    {
        context.ReportDiagnostic(Diagnostic.Create(KernelExpressionBodyNotSupported, expression.GetLocation()));
    }

    public static void ReportReturnTypeNotSupported(TypeSyntax returnType, SourceProductionContext context)
    {
        context.ReportDiagnostic(Diagnostic.Create(ReturnTypeNotSupportedDiagnostic, returnType.GetLocation(), returnType));
    }

    public static void ReportEmptyKernelBody(MethodDeclarationSyntax methodDeclarationSyntax, SourceProductionContext context)
    {
        context.ReportDiagnostic(Diagnostic.Create(EmptyKernelBody, methodDeclarationSyntax.GetLocation()));
    }

    public static void ReportKernelParameterModifiersNotSupported(ParameterSyntax parameter, SourceProductionContext context)
    {
        context.ReportDiagnostic(Diagnostic.Create(KernelParameterModifiersNotSupported, parameter.GetLocation(), parameter));
    }

    public static void ReportParameterTypeNotSupported(TypeSyntax genericNameSyntax, SourceProductionContext context)
    {
        context.ReportDiagnostic(Diagnostic.Create(ParameterTypeNotSupported, genericNameSyntax.GetLocation(), genericNameSyntax));
    }

    public static void ReportUnsupportedStatement(StatementSyntax statement, SourceProductionContext context)
    {
        context.ReportDiagnostic(Diagnostic.Create(StatementTypeNotSupported, statement.GetLocation(), statement));
    }

    public static void ReportInvalidAssignmentExpression(VariableDeclaratorSyntax variable, SourceProductionContext contextContext)
    {
        contextContext.ReportDiagnostic(Diagnostic.Create(InvalidAssignmentExpression, variable.GetLocation(), variable));
    }
}