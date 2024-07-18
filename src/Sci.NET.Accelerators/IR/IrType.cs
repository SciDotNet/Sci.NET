// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sci.NET.Common.Comparison;
using Sci.NET.Common.Numerics;

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// A type in the intermediate representation.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required keyword.")]
public sealed class IrType : IIrWritable
{
    /// <summary>
    /// Gets the void type.
    /// </summary>
    public static IrType Void { get; } = new()
    {
        Name = "void",
        Bits = 0,
        IsIntrinsic = true,
        DotnetType = typeof(void),
        IsPointer = false
    };

    /// <summary>
    /// Gets the pointer type.
    /// </summary>
    [SuppressMessage("Naming", "CA1720:Identifier contains type name", Justification = "This is the name of the type.")]
    public static IrType Pointer { get; } = new()
    {
        Name = "pointer",
        Bits = 0,
        IsIntrinsic = true,
        DotnetType = typeof(void*),
        IsPointer = true,
        Basis = Void
    };

    /// <summary>
    /// Gets the native integer type.
    /// </summary>
    public static IrType NativeInt { get; } = new()
    {
        Name = "nativeint",
        Bits = 0,
        IsIntrinsic = true,
        DotnetType = typeof(nint),
        IsPointer = false
    };

    /// <summary>
    /// Gets the boolean type.
    /// </summary>
    public static IrType Boolean { get; } = new()
    {
        Name = "bool",
        Bits = 1,
        IsIntrinsic = true,
        DotnetType = typeof(bool),
        IsPointer = false
    };

    /// <summary>
    /// Gets the 8-bit integer type.
    /// </summary>
    [SuppressMessage("Naming", "CA1720:Identifier contains type name", Justification = "This is the name of the type.")]
    public static IrType Int8 { get; } = new()
    {
        Name = "int8",
        Bits = 8,
        IsIntrinsic = true,
        DotnetType = typeof(byte),
        IsPointer = false
    };

    /// <summary>
    /// Gets the 16-bit integer type.
    /// </summary>
    [SuppressMessage("Naming", "CA1720:Identifier contains type name", Justification = "This is the name of the type.")]
    public static IrType Int16 { get; } = new()
    {
        Name = "int16",
        Bits = 16,
        IsIntrinsic = true,
        DotnetType = typeof(short),
        IsPointer = false
    };

    /// <summary>
    /// Gets the 32-bit integer type.
    /// </summary>
    [SuppressMessage("Naming", "CA1720:Identifier contains type name", Justification = "This is the name of the type.")]
    public static IrType Int32 { get; } = new()
    {
        Name = "int32",
        Bits = 32,
        IsIntrinsic = true,
        DotnetType = typeof(int),
        IsPointer = false
    };

    /// <summary>
    /// Gets the 64-bit integer type.
    /// </summary>
    [SuppressMessage("Naming", "CA1720:Identifier contains type name", Justification = "This is the name of the type.")]
    public static IrType Int64 { get; } = new()
    {
        Name = "int64",
        Bits = 64,
        IsIntrinsic = true,
        DotnetType = typeof(long),
        IsPointer = false
    };

    /// <summary>
    /// Gets the 128-bit integer type.
    /// </summary>
    public static IrType Int128 { get; } = new()
    {
        Name = "int128",
        Bits = 128,
        IsIntrinsic = true,
        DotnetType = typeof(void*),
        IsPointer = true,
        Basis = Void
    };

    /// <summary>
    /// Gets the half precision floating point type.
    /// </summary>
    public static IrType Fp16 { get; } = new()
    {
        Name = "fp16",
        Bits = 16,
        IsIntrinsic = true,
        DotnetType = typeof(Half),
        IsPointer = false
    };

    /// <summary>
    /// Gets the 16-bit floating point type.
    /// </summary>
    public static IrType Bf16 { get; } = new()
    {
        Name = "bf16",
        Bits = 16,
        IsIntrinsic = true,
        DotnetType = typeof(BFloat16),
        IsPointer = false
    };

    /// <summary>
    /// Gets the single precision floating point type.
    /// </summary>
    [SuppressMessage("Naming", "CA1720:Identifier contains type name", Justification = "This is the name of the type.")]
    public static IrType Fp32 { get; } = new()
    {
        Name = "fp32",
        Bits = 32,
        IsIntrinsic = true,
        DotnetType = typeof(float),
        IsPointer = false
    };

    /// <summary>
    /// Gets the double precision floating point type.
    /// </summary>
    [SuppressMessage("Naming", "CA1720:Identifier contains type name", Justification = "This is the name of the type.")]
    public static IrType Fp64 { get; } = new()
    {
        Name = "fp64",
        Bits = 64,
        IsIntrinsic = true,
        DotnetType = typeof(double),
        IsPointer = false
    };

    /// <summary>
    /// Gets the quadruple precision floating point type.
    /// </summary>
    public static IrType Fp128 { get; } = new()
    {
        Name = "fp128",
        Bits = 128,
        IsIntrinsic = true,
        DotnetType = typeof(void*),
        IsPointer = false,
        Basis = Void
    };

    /// <summary>
    /// Gets the string type.
    /// </summary>
    [SuppressMessage("Naming", "CA1720:Identifier contains type name", Justification = "This is the name of the type.")]
    public static IrType String { get; } = new()
    {
        Name = "string",
        Bits = 0,
        IsIntrinsic = true,
        DotnetType = typeof(string),
        IsPointer = false
    };

    /// <summary>
    /// Gets the character type.
    /// </summary>
    [SuppressMessage("Naming", "CA1720:Identifier contains type name", Justification = "This is the name of the type.")]
    public static IrType Char { get; } = new()
    {
        Name = "char",
        Bits = 16,
        IsIntrinsic = true,
        DotnetType = typeof(char),
        IsPointer = false
    };

    /// <summary>
    /// Gets the name of the type.
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Gets the number of bits in the type.
    /// </summary>
    public required int Bits { get; init; }

    /// <summary>
    /// Gets a value indicating whether the type is intrinsic.
    /// </summary>
    public required bool IsIntrinsic { get; init; }

    /// <summary>
    /// Gets a value indicating whether the type is a pointer.
    /// </summary>
    public required bool IsPointer { get; init; }

    /// <summary>
    /// Gets the basis type.
    /// </summary>
    public IrType? Basis { get; init; }

    /// <summary>
    /// Gets the .NET type.
    /// </summary>
    public required Type DotnetType { get; init; }

    /// <summary>
    /// Makes a pointer to the specified type.
    /// </summary>
    /// <param name="type">The type to make a pointer to.</param>
    /// <returns>The pointer type.</returns>
    public static IrType MakePointer(IrType type)
    {
        return new()
        {
            Name = $"ptr {type.Name}",
            Basis = type,
            Bits = -1,
            IsIntrinsic = false,
            DotnetType = type.DotnetType.MakePointerType(),
            IsPointer = true
        };
    }

    /// <summary>
    /// Creates a default instance of the specified type.
    /// </summary>
    /// <param name="type">The type to create a default instance of.</param>
    /// <returns>The default instance.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the type does not have a default constructor.</exception>
    /// <exception cref="NotSupportedException">Thrown if the type is not a value type.</exception>
    public static ValueType CreateDefaultInstance(IrType type)
    {
        return type.DotnetType switch
        {
            { IsValueType: true } => (ValueType)(Activator.CreateInstance(type.DotnetType) ?? throw new InvalidOperationException()),
            _ => throw new NotSupportedException()
        };
    }

    /// <summary>
    /// Creates a default instance of the specified type.
    /// </summary>
    /// <returns>The default instance.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the type does not have a default constructor.</exception>
    /// <exception cref="NotSupportedException">Thrown if the type is not a value type.</exception>
    public ValueType CreateDefaultInstance()
    {
        return CreateDefaultInstance(this);
    }

    /// <summary>
    /// Makes a pointer to the type.
    /// </summary>
    /// <returns>The pointer type.</returns>
    public IrType MakePointerType()
    {
        return MakePointer(this);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(IrType other)
    {
        return Name == other.Name && Bits == other.Bits && IsIntrinsic == other.IsIntrinsic;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is IrType other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(Name, Bits, IsIntrinsic);
    }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder)
    {
        return builder.Append('@').Append(Name);
    }
}