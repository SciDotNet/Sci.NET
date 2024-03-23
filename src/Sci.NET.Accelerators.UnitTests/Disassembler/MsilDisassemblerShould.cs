// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Diagnostics.SymbolStore;
using System.Reflection;
using System.Reflection.Emit;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.Disassembly.Pdb;

namespace Sci.NET.Accelerators.UnitTests.Disassembler;

public class MsilDisassemblerShould
{
    [Fact]
    public void DisassembleMethod_GivenExample1()
    {
        // Arrange
        var moduleMock = new Mock<Module>();
        var methodBaseMock = new Mock<MethodInfo>();
        var methodBodyMock = new Mock<MethodBody>();
        var leftParameterMock = new Mock<ParameterInfo>();
        var rightParameterMock = new Mock<ParameterInfo>();
        var resultParameterMock = new Mock<ParameterInfo>();
        var lengthParameterMock = new Mock<ParameterInfo>();
        var methodDebugInfoMock = new Mock<MethodDebugInfo>();

        var methodBodyBytes = new byte[]
        {
            0x00, 0x16, 0x0A, 0x2B, 0x1C, 0x00, 0x04, 0x06, 0xD3, 0x1A, 0x5A, 0x58, 0x02, 0x06, 0xD3, 0x1A, 0x5A, 0x58, 0x4E, 0x03, 0x06, 0xD3, 0x1A, 0x5A, 0x58, 0x4E, 0x58, 0x56, 0x00, 0x06, 0x17, 0x58, 0x0A, 0x06, 0x6A, 0x05, 0xFE, 0x04, 0x0B, 0x07, 0x2D, 0xDB, 0x2A
        };

        var variables = new List<LocalVariable> { new () { Index = 0, Name = "i", Type = typeof(long) }, new () { Index = 1, Name = "v_1", Type = typeof(bool) } };
        var parameters = new List<ParameterInfo> { leftParameterMock.Object, rightParameterMock.Object, resultParameterMock.Object, lengthParameterMock.Object };

        methodBaseMock
            .Setup(x => x.GetMethodBody())
            .Returns(methodBodyMock.Object);

        methodBodyMock
            .Setup(x => x.GetILAsByteArray())
            .Returns(methodBodyBytes);

        leftParameterMock.Setup(x => x.Name).Returns("left");
        leftParameterMock.Setup(x => x.ParameterType).Returns(typeof(float*));
        leftParameterMock.Setup(x => x.Position).Returns(0);

        rightParameterMock.Setup(x => x.Name).Returns("right");
        rightParameterMock.Setup(x => x.ParameterType).Returns(typeof(float*));
        rightParameterMock.Setup(x => x.Position).Returns(1);

        resultParameterMock.Setup(x => x.Name).Returns("result");
        resultParameterMock.Setup(x => x.ParameterType).Returns(typeof(float*));
        resultParameterMock.Setup(x => x.Position).Returns(2);

        lengthParameterMock.Setup(x => x.Name).Returns("length");
        lengthParameterMock.Setup(x => x.ParameterType).Returns(typeof(long));
        lengthParameterMock.Setup(x => x.Position).Returns(3);

        methodBaseMock.Setup(x => x.ReturnType).Returns(typeof(void));
        methodBaseMock.Setup(x => x.Name).Returns("Add");

        methodDebugInfoMock.Setup(x => x.LocalScopes).Returns(ImmutableArray<PdbLocalScope>.Empty);
        methodDebugInfoMock.Setup(x => x.SequencePoints).Returns(ImmutableArray<PdbSequencePoint>.Empty);

        var metadata = new MsilMethodMetadata
        {
            Module = moduleMock.Object,
            Parameters = parameters.ToArray(),
            Variables = variables.ToImmutableArray(),
            CodeSize = methodBodyBytes.Length,
            InitLocals = true,
            MaxStack = 5,
            MethodBase = methodBaseMock.Object,
            MethodBody = methodBodyMock.Object,
            ReturnType = typeof(void),
            MethodGenericArguments = Array.Empty<Type>(),
            TypeGenericArguments = Array.Empty<Type>(),
            LocalVariablesSignatureToken = new SymbolToken(999),
            MethodDebugInfo = methodDebugInfoMock.Object
        };

        var disassembler = new MsilDisassembler(metadata);

        // Act
        var disassembledMethod = disassembler.Disassemble();

        // Assert
        var instructions = disassembledMethod.Instructions.ToArray();

        instructions[0].IlOpCode.Should().Be(OpCodeTypes.Nop);
        instructions[1].IlOpCode.Should().Be(OpCodeTypes.Ldc_I4);
        instructions[1].Operand.Should().Be(new MsilInlineIntOperand { Value = 0 });
        instructions[2].IlOpCode.Should().Be(OpCodeTypes.Stloc);
        instructions[2].Operand.Should().Be(new MsilInlineIntOperand { Value = 0 });
        instructions[3].IlOpCode.Should().Be(OpCodeTypes.Br_S);
        instructions[3].Operand.Should().Be(new MsilBranchTargetOperand { Target = 0x21, OperandType = OperandType.ShortInlineBrTarget });
        instructions[4].IlOpCode.Should().Be(OpCodeTypes.Nop);
        instructions[5].IlOpCode.Should().Be(OpCodeTypes.Ldarg);
        instructions[5].Operand.Should().Be(new MsilInlineIntOperand { Value = 2 });
        instructions[6].IlOpCode.Should().Be(OpCodeTypes.Ldloc);
        instructions[6].Operand.Should().Be(new MsilInlineIntOperand { Value = 0 });
        instructions[7].IlOpCode.Should().Be(OpCodeTypes.Conv_I);
        instructions[8].IlOpCode.Should().Be(OpCodeTypes.Ldc_I4);
        instructions[8].Operand.Should().Be(new MsilInlineIntOperand { Value = 4 });
        instructions[9].IlOpCode.Should().Be(OpCodeTypes.Mul);
        instructions[10].IlOpCode.Should().Be(OpCodeTypes.Add);
        instructions[11].IlOpCode.Should().Be(OpCodeTypes.Ldarg);
        instructions[11].Operand.Should().Be(new MsilInlineIntOperand { Value = 0 });
        instructions[12].IlOpCode.Should().Be(OpCodeTypes.Ldloc);
        instructions[12].Operand.Should().Be(new MsilInlineIntOperand { Value = 0 });
        instructions[13].IlOpCode.Should().Be(OpCodeTypes.Conv_I);
        instructions[14].IlOpCode.Should().Be(OpCodeTypes.Ldc_I4);
        instructions[14].Operand.Should().Be(new MsilInlineIntOperand { Value = 4 });
        instructions[15].IlOpCode.Should().Be(OpCodeTypes.Mul);
        instructions[16].IlOpCode.Should().Be(OpCodeTypes.Add);
        instructions[17].IlOpCode.Should().Be(OpCodeTypes.Ldind_R4);
        instructions[18].IlOpCode.Should().Be(OpCodeTypes.Ldarg);
        instructions[18].Operand.Should().Be(new MsilInlineIntOperand { Value = 1 });
        instructions[19].IlOpCode.Should().Be(OpCodeTypes.Ldloc);
        instructions[19].Operand.Should().Be(new MsilInlineIntOperand { Value = 0 });
        instructions[20].IlOpCode.Should().Be(OpCodeTypes.Conv_I);
        instructions[21].IlOpCode.Should().Be(OpCodeTypes.Ldc_I4);
        instructions[21].Operand.Should().Be(new MsilInlineIntOperand { Value = 4 });
        instructions[22].IlOpCode.Should().Be(OpCodeTypes.Mul);
        instructions[23].IlOpCode.Should().Be(OpCodeTypes.Add);
        instructions[24].IlOpCode.Should().Be(OpCodeTypes.Ldind_R4);
        instructions[25].IlOpCode.Should().Be(OpCodeTypes.Add);
        instructions[26].IlOpCode.Should().Be(OpCodeTypes.Stind_R4);
        instructions[27].IlOpCode.Should().Be(OpCodeTypes.Nop);
        instructions[28].IlOpCode.Should().Be(OpCodeTypes.Ldloc);
        instructions[29].IlOpCode.Should().Be(OpCodeTypes.Ldc_I4);
        instructions[29].Operand.Should().Be(new MsilInlineIntOperand { Value = 1 });
        instructions[30].IlOpCode.Should().Be(OpCodeTypes.Add);
        instructions[31].IlOpCode.Should().Be(OpCodeTypes.Stloc);
        instructions[31].Operand.Should().Be(new MsilInlineIntOperand { Value = 0 });
        instructions[32].IlOpCode.Should().Be(OpCodeTypes.Ldloc);
        instructions[32].Operand.Should().Be(new MsilInlineIntOperand { Value = 0 });
        instructions[33].IlOpCode.Should().Be(OpCodeTypes.Conv_I8);
        instructions[34].IlOpCode.Should().Be(OpCodeTypes.Ldarg);
        instructions[34].Operand.Should().Be(new MsilInlineIntOperand { Value = 3 });
        instructions[35].IlOpCode.Should().Be(OpCodeTypes.Clt);
        instructions[36].IlOpCode.Should().Be(OpCodeTypes.Stloc);
        instructions[36].Operand.Should().Be(new MsilInlineIntOperand { Value = 1 });
        instructions[37].IlOpCode.Should().Be(OpCodeTypes.Ldloc);
        instructions[37].Operand.Should().Be(new MsilInlineIntOperand { Value = 1 });
        instructions[38].IlOpCode.Should().Be(OpCodeTypes.Brtrue_S);
        instructions[38].Operand.Should().Be(new MsilBranchTargetOperand { Target = 0x05, OperandType = OperandType.ShortInlineBrTarget });
        instructions[39].IlOpCode.Should().Be(OpCodeTypes.Ret);
    }
}