// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.SymbolStore;
using System.Reflection;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Rewriter;

namespace Sci.NET.Accelerators.UnitTests.Rewriter;

public class MsilToIrTranslatorShould
{
    [Fact]
    public void NotThrow()
    {
        // Arrange
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

        methodBaseMock.Setup(x => x.ReturnType).Returns(typeof(void));
        methodBaseMock.Setup(x => x.Name).Returns("Add");

        var metadata = new MsilMethodMetadata
        {
            Module = moduleMock.Object,
            Parameters = parameters.ToArray(),
            Variables = variables.ToArray(),
            CodeSize = methodBodyBytes.Length,
            InitLocals = true,
            MaxStack = 5,
            MethodBase = methodBaseMock.Object,
            MethodBody = methodBodyMock.Object,
            ReturnType = typeof(void),
            MethodGenericArguments = Array.Empty<Type>(),
            TypeGenericArguments = Array.Empty<Type>(),
            LocalVariablesSignatureToken = new SymbolToken(999)
        };

        var disassembler = new MsilDisassembler(metadata);
        var disassembledMethod = disassembler.Disassemble();
        var ssaTransformer = new MsilToIrTranslator(disassembledMethod);

        // Act
        var ssaMethod = ssaTransformer.Transform();

        // Assert
        Assert.NotNull(ssaMethod);
    }
}