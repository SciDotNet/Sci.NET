// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.Disassembly;

/// <summary>
/// Represents a pop behaviour.
/// </summary>
[Flags]
public enum PopBehaviour
{
#pragma warning disable CS1591 // Missing XML comment for publicly visible type or member
#pragma warning disable SA1602
#pragma warning disable CA1707
    None = 0,
    Pop1 = 1,
    Pop1_pop1 = 1 << 1,
    Popi_pop1 = 1 << 2,
    Popi_popr4 = 1 << 3,
    Popref_popi_popr8 = 1 << 4,
    Popi = Pop1 | Pop1_pop1,
    Popi_popi = Pop1 | Popi_pop1,
    Popi_popi8 = Pop1_pop1 | Popi_pop1,
    Popi_popi_popi = Popi_popi8 | Pop1,
    Popi_popr8 = Pop1 | Popi_popr4,
    Popref = Pop1_pop1 | Popi_popr4,
    Popref_pop1 = Popref | Pop1,
    Popref_popi = Popi_pop1 | Popi_popr4,
    Popref_popi_popi = Popref_popi | Pop1,
    Popref_popi_popi8 = Popref_popi | Pop1_pop1,
    Popref_popi_popr4 = Popref_popi_popi8 | Pop1,
    Popref_popi_popref = Pop1 | Popref_popi_popr8,
    Popref_popi_pop1 = Popref_popi | Popref_popi_popr8,
    Varpop = Popref | Popref_popi_popr8
}
#pragma warning restore CS1591 // Missing XML comment for publicly visible type or member
#pragma warning restore SA1602
#pragma warning restore CA1707