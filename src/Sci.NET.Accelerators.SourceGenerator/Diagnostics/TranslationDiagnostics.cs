// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Sci.NET.Accelerators.SourceGenerator.Diagnostics;

/// <summary>
/// A collection of diagnostics for the translation process.
/// </summary>
[PublicAPI]
public static class TranslationDiagnostics
{
    private static readonly DiagnosticDescriptor MissingBody = new (
        id: "SCI0001",
        title: new LocalizableResourceString(nameof(Resources.SCI0001Title), Resources.ResourceManager, typeof(Resources)),
        messageFormat: new LocalizableResourceString(nameof(Resources.SCI0001MessageFormat), Resources.ResourceManager, typeof(Resources)),
        category: "Translation",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor UnsupportedParameterModifier = new (
        id: "SCI0002",
        title: new LocalizableResourceString(nameof(Resources.SCI0002Title), Resources.ResourceManager, typeof(Resources)),
        messageFormat: new LocalizableResourceString(nameof(Resources.SCI0002MessageFormat), Resources.ResourceManager, typeof(Resources)),
        category: "Translation",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor UnsupportedParameterType = new (
        id: "SCI0003",
        title: new LocalizableResourceString(nameof(Resources.SCI0003Title), Resources.ResourceManager, typeof(Resources)),
        messageFormat: new LocalizableResourceString(nameof(Resources.SCI0003MessageFormat), Resources.ResourceManager, typeof(Resources)),
        category: "Translation",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    /// <summary>
    /// Reports a missing body for a method.
    /// </summary>
    /// <param name="context">The source production context.</param>
    /// <param name="methodDeclarationSyntax">The method declaration syntax which is missing a body.</param>
    public static void ReportMissingBody(SourceProductionContext context, MethodDeclarationSyntax methodDeclarationSyntax)
    {
        var diagnostic = Diagnostic.Create(MissingBody, methodDeclarationSyntax.Identifier.GetLocation(), methodDeclarationSyntax.Identifier.ValueText);
        context.ReportDiagnostic(diagnostic);
    }

    /// <summary>
    /// Reports an unsupported parameter modifier.
    /// </summary>
    /// <param name="translationContextSourceProductionContext">The translation context source production context.</param>
    /// <param name="parameter">The parameter.</param>
    public static void ReportUnsupportedParameterModifier(SourceProductionContext translationContextSourceProductionContext, ParameterSyntax parameter)
    {
        var diagnostic = Diagnostic.Create(UnsupportedParameterModifier, parameter.Identifier.GetLocation(), parameter.Identifier.ValueText);
        translationContextSourceProductionContext.ReportDiagnostic(diagnostic);
    }

    /// <summary>
    /// Reports an unsupported parameter type.
    /// </summary>
    /// <param name="context">The translation context source production context.</param>
    /// <param name="parameter">The parameter.</param>
    public static void ReportUnsupportedParameterType(SourceProductionContext context, ParameterSyntax parameter)
    {
        var diagnostic = Diagnostic.Create(UnsupportedParameterType, parameter.Identifier.GetLocation(), parameter.Identifier.ValueText);
        context.ReportDiagnostic(diagnostic);
    }
}