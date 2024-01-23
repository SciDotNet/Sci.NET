// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace Sci.NET.Accelerators.Rewriter.Variables;

/// <summary>
/// Represents a variable name generator.
/// </summary>
[PublicAPI]
[SuppressMessage("Design", "CA1024:Use properties where appropriate", Justification = "Should be methods.")]
public class VariableNameGenerator
{
    private int _localCount;
    private int _temporaryCount;
    private int _constantCount;

    /// <summary>
    /// Initializes a new instance of the <see cref="VariableNameGenerator"/> class.
    /// </summary>
    public VariableNameGenerator()
    {
        _localCount = 0;
        _temporaryCount = 0;
        _constantCount = 0;
    }

    /// <summary>
    /// Gets a local name.
    /// </summary>
    /// <returns>The local name.</returns>
    public string NextLocalName()
    {
        return $"loc_{_localCount++}";
    }

    /// <summary>
    /// Gets a temporary name.
    /// </summary>
    /// <returns>The temporary name.</returns>
    public string NextTemporaryName()
    {
        return $"tmp_{_temporaryCount++}";
    }

    /// <summary>
    /// Gets a constant name.
    /// </summary>
    /// <returns>The constant name.</returns>
    public string NextConstant()
    {
        return $"const_{_constantCount++}";
    }

    /// <summary>
    /// Gets the next temporary variable.
    /// </summary>
    /// <param name="type">The type.</param>
    /// <returns>The temporary variable.</returns>
    public ISsaVariable GetNextTemp(Type type)
    {
        return new TempSsaVariable(NextTemporaryName(), type);
    }
}