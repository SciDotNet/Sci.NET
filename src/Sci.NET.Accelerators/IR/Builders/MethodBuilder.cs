// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.IR.Builders;

/// <summary>
/// Provides methods to build methods in the IR.
/// </summary>
[PublicAPI]
public class MethodBuilder
{
    private readonly List<Parameter> _parameters;
    private readonly List<LocalVariable> _localVariables;

    /// <summary>
    /// Initializes a new instance of the <see cref="MethodBuilder"/> class.
    /// </summary>
    public MethodBuilder()
    {
        _parameters = new List<Parameter>();
        _localVariables = new List<LocalVariable>();
    }

    /// <summary>
    /// Adds a parameter to the method.
    /// </summary>
    /// <param name="parameter">The parameter to add.</param>
    public void AddParameter(Parameter parameter)
    {
        _parameters.Add(parameter);
    }

    /// <summary>
    /// Adds a local variable to the method.
    /// </summary>
    /// <param name="localVariable">The local variable to add.</param>
    public void AddLocalVariable(LocalVariable localVariable)
    {
        _localVariables.Add(localVariable);
    }
}