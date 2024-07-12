// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Diagnostics.SymbolStore;
using System.Reflection;
using System.Reflection.Emit;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Pdb;
using Sci.NET.Accelerators.IR;
using Sci.NET.Accelerators.IR.Instructions;
using Sci.NET.Accelerators.IR.Instructions.Arithmetic;
using Sci.NET.Accelerators.IR.Instructions.Comparison;
using Sci.NET.Accelerators.IR.Instructions.Conversion;
using Sci.NET.Accelerators.IR.Instructions.MemoryAccess;
using Sci.NET.Accelerators.IR.Instructions.Terminators;
using Sci.NET.Accelerators.Rewriter;

namespace Sci.NET.Accelerators.UnitTests.Rewriter;

public class MsilToIrTranslatorShould
{
    [Fact]
    public void ReturnCorrectInstructions_GivenSampleMethodBody()
    {
        // Arrange
        var moduleMock = new Mock<Module>();
        var methodBaseMock = new Mock<MethodInfo>();
        var methodBodyMock = new Mock<MethodBody>();
        var leftParameterMock = new Mock<ParameterInfo>();
        var rightParameterMock = new Mock<ParameterInfo>();
        var resultParameterMock = new Mock<ParameterInfo>();
        var lengthParameterMock = new Mock<ParameterInfo>();

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

        methodBaseMock.Setup(x => x.ReturnType).Returns(typeof(long));
        methodBaseMock.Setup(x => x.Name).Returns("Add");

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
            ReturnType = typeof(long),
            MethodGenericArguments = Array.Empty<Type>(),
            TypeGenericArguments = Array.Empty<Type>(),
            LocalVariablesSignatureToken = new SymbolToken(999),
            MethodDebugInfo = new MethodDebugInfo
            {
                Handle = default,
                MethodBase = new DynamicMethod("Add", typeof(long), new[] { typeof(float*), typeof(float*), typeof(float*), typeof(long) }),
                AssemblyDebugInformation = new FakeAssemblyDebugInformation(Assembly.GetExecutingAssembly())
            }
        };

        var disassembler = new MsilDisassembler(metadata);
        var disassembledMethod = disassembler.Disassemble();
        var ssaTransformer = new MsilToIrTranslator(disassembledMethod);

        // Act
        var ssaMethod = ssaTransformer.Transform();

        // Assert
        ssaMethod.Locals.Count.Should().Be(variables.Count);
        ssaMethod.Parameters.Count.Should().Be(parameters.Count);
        ssaMethod.ReturnType.Should().Be(typeof(long));
        ssaMethod.BasicBlocks.Count.Should().Be(4);

        var blocks = ssaMethod.BasicBlocks.ToArray();

        blocks.Skip(1).Sum(x => x.Instructions.Count).Should().Be(disassembledMethod.Instructions.Length);

        AssertBlock0(blocks[0], blocks);
        AssertBlock1(blocks[1], blocks);
        AssertBlock2(blocks[2], blocks);
        AssertBlock3(blocks[3], blocks);
        AssertBlock4(blocks[4]);
    }

    private static void AssertBlock0(BasicBlock firstBlock, BasicBlock[] blocks)
    {
        firstBlock.Name.Should().Be("block_0");
        firstBlock.Instructions.Count.Should().Be(3);

        firstBlock.Instructions[0].Should().BeOfType<DeclareLocalInstruction>();
        var instruction1 = firstBlock.Instructions[0] as DeclareLocalInstruction;
        instruction1.Should().NotBeNull();
        instruction1?.Result.Identifier.Should().Be("loc_0");
        instruction1?.Result.Type.Should().Be(IrType.Int64);

        firstBlock.Instructions[1].Should().BeOfType<DeclareLocalInstruction>();
        var instruction2 = firstBlock.Instructions[1] as DeclareLocalInstruction;
        instruction2.Should().NotBeNull();
        instruction2?.Result.Identifier.Should().Be("loc_1");
        instruction2?.Result.Type.Should().Be(IrType.Boolean);

        firstBlock.Instructions[2].Should().BeOfType<BranchInstruction>();
        var instruction3 = firstBlock.Instructions[2] as BranchInstruction;
        instruction3.Should().NotBeNull();
        instruction3?.Target.Should().Be(blocks[1]);
    }

    private static void AssertBlock1(BasicBlock block, BasicBlock[] blocks)
    {
        block.Name.Should().Be("block_1");
        block.Instructions.Count.Should().Be(4);

        block.Instructions[0].Should().BeOfType<NopInstruction>();

        block.Instructions[1].Should().BeOfType<LoadConstantInstruction>();
        var instruction1 = block.Instructions[1] as LoadConstantInstruction;
        instruction1.Should().NotBeNull();
        instruction1?.Result.Identifier.Should().Be("tmp_1");
        instruction1?.Result.Type.Should().Be(IrType.Int32);

        block.Instructions[2].Should().BeOfType<StoreLocalInstruction>();
        var instruction2 = block.Instructions[2] as StoreLocalInstruction;
        instruction2.Should().NotBeNull();
        instruction2?.Local.Identifier.Should().Be("loc_0");
        instruction2?.Local.Type.Should().Be(IrType.Int64);
        instruction2?.Value.Identifier.Should().Be("tmp_1");
        instruction2?.Value.Type.Should().Be(IrType.Int32);

        block.Instructions[3].Should().BeOfType<BranchInstruction>();
        var instruction3 = block.Instructions[3] as BranchInstruction;
        instruction3.Should().NotBeNull();
        instruction3?.Target.Should().Be(blocks[3]);
    }

    private static void AssertBlock2(BasicBlock block, BasicBlock[] blocks)
    {
        block.Name.Should().Be("block_2");
        block.Instructions.Count.Should().Be(29);

        block.Instructions[0].Should().BeOfType<NopInstruction>();

        block.Instructions[1].Should().BeOfType<LoadArgumentInstruction>();
        var instruction0 = block.Instructions[1] as LoadArgumentInstruction;
        instruction0.Should().NotBeNull();
        instruction0?.Result.Identifier.Should().Be("tmp_2");
        instruction0?.Result.Type.Should().Be(IrType.MakePointer(IrType.Fp32));
        instruction0?.Parameter.Identifier.Should().Be("arg_result");

        block.Instructions[2].Should().BeOfType<LoadLocalInstruction>();
        var instruction1 = block.Instructions[2] as LoadLocalInstruction;
        instruction1.Should().NotBeNull();
        instruction1?.Result.Identifier.Should().Be("tmp_3");
        instruction1?.Result.Type.Should().Be(IrType.Int64);
        instruction1?.Local.Identifier.Should().Be("loc_0");

        block.Instructions[3].Should().BeOfType<SignExtendInstruction>();
        var instruction2 = block.Instructions[3] as SignExtendInstruction;
        instruction2.Should().NotBeNull();
        instruction2?.Result.Identifier.Should().Be("tmp_4");
        instruction2?.Value.Identifier.Should().Be("tmp_3");
        instruction2?.Result.Type.Should().Be(IrType.NativeInt);

        block.Instructions[4].Should().BeOfType<LoadConstantInstruction>();
        var instruction3 = block.Instructions[4] as LoadConstantInstruction;
        instruction3.Should().NotBeNull();
        instruction3?.Result.Identifier.Should().Be("tmp_6");
        instruction3?.Result.Type.Should().Be(IrType.Int32);
        instruction3?.Value.Should().Be(4);

        block.Instructions[5].Should().BeOfType<MultiplyInstruction>();
        var instruction4 = block.Instructions[5] as MultiplyInstruction;
        instruction4.Should().NotBeNull();
        instruction4?.Result.Identifier.Should().Be("tmp_7");
        instruction4?.Left.Identifier.Should().Be("tmp_4");
        instruction4?.Right.Identifier.Should().Be("tmp_6");
        instruction4?.Result.Type.Should().Be(IrType.NativeInt);
        instruction4?.Left.Type.Should().Be(IrType.NativeInt);
        instruction4?.Right.Type.Should().Be(IrType.Int32);

        block.Instructions[6].Should().BeOfType<AddInstruction>();
        var instruction5 = block.Instructions[6] as AddInstruction;
        instruction5.Should().NotBeNull();
        instruction5?.Result.Identifier.Should().Be("tmp_8");
        instruction5?.Left.Identifier.Should().Be("tmp_2");
        instruction5?.Right.Identifier.Should().Be("tmp_7");
        instruction5?.Result.Type.Should().Be(IrType.MakePointer(IrType.Fp32));
        instruction5?.Left.Type.Should().Be(IrType.MakePointer(IrType.Fp32));
        instruction5?.Right.Type.Should().Be(IrType.NativeInt);

        block.Instructions[7].Should().BeOfType<LoadArgumentInstruction>();
        var instruction6 = block.Instructions[7] as LoadArgumentInstruction;
        instruction6.Should().NotBeNull();
        instruction6?.Result.Identifier.Should().Be("tmp_9");
        instruction6?.Result.Type.Should().Be(IrType.MakePointer(IrType.Fp32));
        instruction6?.Parameter.Identifier.Should().Be("arg_left");

        block.Instructions[8].Should().BeOfType<LoadLocalInstruction>();
        var instruction7 = block.Instructions[8] as LoadLocalInstruction;
        instruction7.Should().NotBeNull();
        instruction7?.Result.Identifier.Should().Be("tmp_10");
        instruction7?.Result.Type.Should().Be(IrType.Int64);
        instruction7?.Local.Identifier.Should().Be("loc_0");

        block.Instructions[9].Should().BeOfType<SignExtendInstruction>();
        var instruction8 = block.Instructions[9] as SignExtendInstruction;
        instruction8.Should().NotBeNull();
        instruction8?.Result.Identifier.Should().Be("tmp_11");
        instruction8?.Value.Identifier.Should().Be("tmp_10");
        instruction8?.Result.Type.Should().Be(IrType.NativeInt);
        instruction8?.Value.Type.Should().Be(IrType.Int64);

        block.Instructions[10].Should().BeOfType<LoadConstantInstruction>();
        var instruction9 = block.Instructions[10] as LoadConstantInstruction;
        instruction9.Should().NotBeNull();
        instruction9?.Result.Identifier.Should().Be("tmp_13");
        instruction9?.Result.Type.Should().Be(IrType.Int32);
        instruction9?.Value.Should().Be(4);

        block.Instructions[11].Should().BeOfType<MultiplyInstruction>();
        var instruction10 = block.Instructions[11] as MultiplyInstruction;
        instruction10.Should().NotBeNull();
        instruction10?.Result.Identifier.Should().Be("tmp_14");
        instruction10?.Left.Identifier.Should().Be("tmp_11");
        instruction10?.Right.Identifier.Should().Be("tmp_13");
        instruction10?.Result.Type.Should().Be(IrType.NativeInt);
        instruction10?.Left.Type.Should().Be(IrType.NativeInt);
        instruction10?.Right.Type.Should().Be(IrType.Int32);

        block.Instructions[12].Should().BeOfType<AddInstruction>();
        var instruction11 = block.Instructions[12] as AddInstruction;
        instruction11.Should().NotBeNull();
        instruction11?.Result.Identifier.Should().Be("tmp_15");
        instruction11?.Left.Identifier.Should().Be("tmp_9");
        instruction11?.Right.Identifier.Should().Be("tmp_14");
        instruction11?.Result.Type.Should().Be(IrType.MakePointer(IrType.Fp32));
        instruction11?.Left.Type.Should().Be(IrType.MakePointer(IrType.Fp32));
        instruction11?.Right.Type.Should().Be(IrType.NativeInt);

        block.Instructions[13].Should().BeOfType<LoadElementFromPointerInstruction>();
        var instruction12 = block.Instructions[13] as LoadElementFromPointerInstruction;
        instruction12.Should().NotBeNull();
        instruction12?.Result.Identifier.Should().Be("tmp_16");
        instruction12?.Result.Type.Should().Be(IrType.Fp32);
        instruction12?.Pointer.Identifier.Should().Be("tmp_15");
        instruction12?.Pointer.Type.Should().Be(IrType.MakePointer(IrType.Fp32));

        block.Instructions[14].Should().BeOfType<LoadArgumentInstruction>();
        var instruction13 = block.Instructions[14] as LoadArgumentInstruction;
        instruction13.Should().NotBeNull();
        instruction13?.Result.Identifier.Should().Be("tmp_17");
        instruction13?.Result.Type.Should().Be(IrType.MakePointer(IrType.Fp32));
        instruction13?.Parameter.Identifier.Should().Be("arg_right");

        block.Instructions[15].Should().BeOfType<LoadLocalInstruction>();
        var instruction14 = block.Instructions[15] as LoadLocalInstruction;
        instruction14.Should().NotBeNull();
        instruction14?.Result.Identifier.Should().Be("tmp_18");
        instruction14?.Result.Type.Should().Be(IrType.Int64);
        instruction14?.Local.Identifier.Should().Be("loc_0");

        block.Instructions[16].Should().BeOfType<SignExtendInstruction>();
        var instruction15 = block.Instructions[16] as SignExtendInstruction;
        instruction15.Should().NotBeNull();
        instruction15?.Result.Identifier.Should().Be("tmp_19");
        instruction15?.Value.Identifier.Should().Be("tmp_18");
        instruction15?.Result.Type.Should().Be(IrType.NativeInt);
        instruction15?.Value.Type.Should().Be(IrType.Int64);

        block.Instructions[17].Should().BeOfType<LoadConstantInstruction>();
        var instruction16 = block.Instructions[17] as LoadConstantInstruction;
        instruction16.Should().NotBeNull();
        instruction16?.Result.Identifier.Should().Be("tmp_21");
        instruction16?.Result.Type.Should().Be(IrType.Int32);
        instruction16?.Value.Should().Be(4);

        block.Instructions[18].Should().BeOfType<MultiplyInstruction>();
        var instruction17 = block.Instructions[18] as MultiplyInstruction;
        instruction17.Should().NotBeNull();
        instruction17?.Result.Identifier.Should().Be("tmp_22");
        instruction17?.Left.Identifier.Should().Be("tmp_19");
        instruction17?.Right.Identifier.Should().Be("tmp_21");
        instruction17?.Result.Type.Should().Be(IrType.NativeInt);
        instruction17?.Left.Type.Should().Be(IrType.NativeInt);
        instruction17?.Right.Type.Should().Be(IrType.Int32);

        block.Instructions[19].Should().BeOfType<AddInstruction>();
        var instruction18 = block.Instructions[19] as AddInstruction;
        instruction18.Should().NotBeNull();
        instruction18?.Result.Identifier.Should().Be("tmp_23");
        instruction18?.Left.Identifier.Should().Be("tmp_17");
        instruction18?.Right.Identifier.Should().Be("tmp_22");
        instruction18?.Result.Type.Should().Be(IrType.MakePointer(IrType.Fp32));
        instruction18?.Left.Type.Should().Be(IrType.MakePointer(IrType.Fp32));
        instruction18?.Right.Type.Should().Be(IrType.NativeInt);

        block.Instructions[20].Should().BeOfType<LoadElementFromPointerInstruction>();
        var instruction19 = block.Instructions[20] as LoadElementFromPointerInstruction;
        instruction19.Should().NotBeNull();
        instruction19?.Result.Identifier.Should().Be("tmp_24");
        instruction19?.Result.Type.Should().Be(IrType.Fp32);
        instruction19?.Pointer.Identifier.Should().Be("tmp_23");
        instruction19?.Pointer.Type.Should().Be(IrType.MakePointer(IrType.Fp32));

        block.Instructions[21].Should().BeOfType<AddInstruction>();
        var instruction20 = block.Instructions[21] as AddInstruction;
        instruction20.Should().NotBeNull();
        instruction20?.Result.Identifier.Should().Be("tmp_25");
        instruction20?.Left.Identifier.Should().Be("tmp_16");
        instruction20?.Right.Identifier.Should().Be("tmp_24");
        instruction20?.Result.Type.Should().Be(IrType.Fp32);
        instruction20?.Left.Type.Should().Be(IrType.Fp32);
        instruction20?.Right.Type.Should().Be(IrType.Fp32);

        block.Instructions[22].Should().BeOfType<StoreElementAtPointerInstruction>();
        var instruction21 = block.Instructions[22] as StoreElementAtPointerInstruction;
        instruction21.Should().NotBeNull();
        instruction21?.Pointer.Identifier.Should().Be("tmp_25");
        instruction21?.Pointer.Type.Should().Be(IrType.Fp32);
        instruction21?.Pointer.Identifier.Should().Be("tmp_25");
        instruction21?.Value.Identifier.Should().Be("tmp_8");
        instruction21?.Value.Type.Should().Be(IrType.MakePointer(IrType.Fp32));

        block.Instructions[23].Should().BeOfType<NopInstruction>();

        block.Instructions[24].Should().BeOfType<LoadLocalInstruction>();
        var instruction22 = block.Instructions[24] as LoadLocalInstruction;
        instruction22.Should().NotBeNull();
        instruction22?.Result.Identifier.Should().Be("tmp_26");
        instruction22?.Result.Type.Should().Be(IrType.Int64);
        instruction22?.Local.Identifier.Should().Be("loc_0");

        block.Instructions[25].Should().BeOfType<LoadConstantInstruction>();
        var instruction23 = block.Instructions[25] as LoadConstantInstruction;
        instruction23.Should().NotBeNull();
        instruction23?.Result.Identifier.Should().Be("tmp_28");
        instruction23?.Result.Type.Should().Be(IrType.Int32);
        instruction23?.Value.Should().Be(1);

        block.Instructions[26].Should().BeOfType<AddInstruction>();
        var instruction24 = block.Instructions[26] as AddInstruction;
        instruction24.Should().NotBeNull();
        instruction24?.Result.Identifier.Should().Be("tmp_29");
        instruction24?.Left.Identifier.Should().Be("tmp_26");
        instruction24?.Right.Identifier.Should().Be("tmp_28");
        instruction24?.Result.Type.Should().Be(IrType.Int64);
        instruction24?.Left.Type.Should().Be(IrType.Int64);
        instruction24?.Right.Type.Should().Be(IrType.Int32);

        block.Instructions[27].Should().BeOfType<StoreLocalInstruction>();
        var instruction25 = block.Instructions[27] as StoreLocalInstruction;
        instruction25.Should().NotBeNull();
        instruction25?.Local.Identifier.Should().Be("loc_0");
        instruction25?.Local.Type.Should().Be(IrType.Int64);
        instruction25?.Value.Identifier.Should().Be("tmp_29");
        instruction25?.Value.Type.Should().Be(IrType.Int64);

        block.Instructions[28].Should().BeOfType<BranchInstruction>();
        var instruction29 = block.Instructions[28] as BranchInstruction;
        instruction29.Should().NotBeNull();
        instruction29?.Target.Should().Be(blocks[3]);
    }

    private static void AssertBlock3(BasicBlock block, BasicBlock[] blocks)
    {
        block.Name.Should().Be("block_3");
        block.Instructions.Count.Should().Be(7);

        block.Instructions[0].Should().BeOfType<LoadLocalInstruction>();
        var instruction1 = block.Instructions[0] as LoadLocalInstruction;
        instruction1.Should().NotBeNull();
        instruction1?.Result.Identifier.Should().Be("tmp_30");
        instruction1?.Result.Type.Should().Be(IrType.Int64);
        instruction1?.Local.Identifier.Should().Be("loc_0");

        block.Instructions[1].Should().BeOfType<SignExtendInstruction>();
        var instruction2 = block.Instructions[1] as SignExtendInstruction;
        instruction2.Should().NotBeNull();
        instruction2?.Result.Identifier.Should().Be("tmp_31");
        instruction2?.Value.Identifier.Should().Be("tmp_30");
        instruction2?.Result.Type.Should().Be(IrType.Int64);
        instruction2?.Value.Type.Should().Be(IrType.Int64);

        block.Instructions[2].Should().BeOfType<LoadArgumentInstruction>();
        var instruction3 = block.Instructions[2] as LoadArgumentInstruction;
        instruction3.Should().NotBeNull();
        instruction3?.Result.Identifier.Should().Be("tmp_32");
        instruction3?.Result.Type.Should().Be(IrType.Int64);
        instruction3?.Parameter.Identifier.Should().Be("arg_length");

        block.Instructions[3].Should().BeOfType<CompareLessThanInstruction>();
        var instruction4 = block.Instructions[3] as CompareLessThanInstruction;
        instruction4.Should().NotBeNull();
        instruction4?.Result.Identifier.Should().Be("tmp_33");
        instruction4?.Left.Identifier.Should().Be("tmp_31");
        instruction4?.Right.Identifier.Should().Be("tmp_32");
        instruction4?.Result.Type.Should().Be(IrType.Boolean);
        instruction4?.Left.Type.Should().Be(IrType.Int64);
        instruction4?.Right.Type.Should().Be(IrType.Int64);

        block.Instructions[4].Should().BeOfType<StoreLocalInstruction>();
        var instruction5 = block.Instructions[4] as StoreLocalInstruction;
        instruction5.Should().NotBeNull();
        instruction5?.Local.Identifier.Should().Be("loc_1");
        instruction5?.Local.Type.Should().Be(IrType.Boolean);
        instruction5?.Value.Identifier.Should().Be("tmp_33");
        instruction5?.Value.Type.Should().Be(IrType.Boolean);

        block.Instructions[5].Should().BeOfType<LoadLocalInstruction>();
        var instruction6 = block.Instructions[5] as LoadLocalInstruction;
        instruction6.Should().NotBeNull();
        instruction6?.Result.Identifier.Should().Be("tmp_34");
        instruction6?.Result.Type.Should().Be(IrType.Boolean);
        instruction6?.Local.Identifier.Should().Be("loc_1");

        block.Instructions[6].Should().BeOfType<ConditionalBranchInstruction>();
        var instruction7 = block.Instructions[6] as ConditionalBranchInstruction;
        instruction7.Should().NotBeNull();
        instruction7?.Condition.Identifier.Should().Be("tmp_34");
        instruction7?.Condition.Type.Should().Be(IrType.Boolean);
        instruction7?.Target.Should().Be(blocks[2]);
        instruction7?.FalseTarget.Should().Be(blocks[4]);
    }

    private static void AssertBlock4(BasicBlock block)
    {
        block.Name.Should().Be("block_4");
        block.Instructions.Count.Should().Be(1);

        block.Instructions[0].Should().BeOfType<ReturnInstruction>();
        var instruction1 = block.Instructions[0] as ReturnInstruction;
        instruction1.Should().NotBeNull();
        instruction1?.Value.Identifier.Should().Be("tmp_30");
        instruction1?.Value.Type.Should().Be(IrType.Int64);
    }
}