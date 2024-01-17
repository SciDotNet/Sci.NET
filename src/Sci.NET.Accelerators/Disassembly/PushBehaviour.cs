// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.Disassembly;

/// <summary>
/// Represents a MSIL instruction push behaviour.
/// </summary>
[PublicAPI]
public enum PushBehaviour
{
#pragma warning disable CS1591, SA1602, CA1707
    None = 0,
    Varpush = 27,
    Push0 = 18,
    Push1 = 19,
    Push1_push1 = 20,
    Pushi = 21,
    Pushi8 = 22,
    Pushr4 = 23,
    Pushr8 = 24,
    Pushref = 25,
#pragma warning restore CS1591, SA1602, CA1707
}