// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.IR;

namespace Sci.NET.Accelerators.Rewriter;

internal class CfgBuilder
{
    private readonly Dictionary<int, int> _instructionOffsetMapping;
    private readonly Dictionary<int, int> _basicBlockMapping;
    private readonly Dictionary<int, List<BasicBlock>> _successorMapping;
    private readonly Dictionary<int, BasicBlock> _basicBlocks;
    private readonly DisassembledMsilMethod _disassembledMsilMethod;

    public CfgBuilder(DisassembledMsilMethod disassembledMsilMethod)
    {
        _instructionOffsetMapping = new Dictionary<int, int>();
        _basicBlockMapping = new Dictionary<int, int>();
        _successorMapping = new Dictionary<int, List<BasicBlock>>();
        _basicBlocks = new Dictionary<int, BasicBlock>();
        _disassembledMsilMethod = disassembledMsilMethod;
    }

    public List<BasicBlock> Build()
    {
        CreateInstructionOffsetMapping();
        FindBasicBlockMapping();
        ConstructBasicBlocks();

        var instructions = _disassembledMsilMethod.Instructions;
        var cfgInstructions = _basicBlocks.Values.SelectMany(x => x.MsilInstructions).ToList();

        Debug.Assert(_basicBlocks.Values.Sum(x => x.MsilInstructions.Count) == _disassembledMsilMethod.Instructions.Length, "_basicBlocks.Values.Sum(x => x.MsilInstructions) == _disassembledMsilMethod.Instructions.Length");

        ComputeSuccessors();
        AddPredecessorsAndSuccessors();

        return _basicBlocks.Values.ToList();
    }

    private void AddPredecessorsAndSuccessors()
    {
        foreach (var (index, successors) in _successorMapping)
        {
            foreach (var successor in successors)
            {
                _basicBlocks[index].AddSuccessor(successor);
                successor.AddPredecessor(_basicBlocks[index]);
            }
        }
    }

    private void ComputeSuccessors()
    {
        foreach (var basicBlock in _basicBlocks.SkipLast(1))
        {
            var lastInstruction = basicBlock.Value.MsilInstructions.Last();

            if (!_successorMapping.ContainsKey(lastInstruction.Offset))
            {
                _successorMapping.Add(basicBlock.Key, new List<BasicBlock>());
            }

            if (lastInstruction.IsBranch)
            {
                var branchTargets = lastInstruction.GetBranchTargets();

                _successorMapping[basicBlock.Key].AddRange(branchTargets.Select(x => _basicBlocks[x]).ToList());
            }
            else
            {
                var nextInstructionIndex = _instructionOffsetMapping[lastInstruction.Offset] + lastInstruction.Size;
                _successorMapping[basicBlock.Key].Add(_basicBlocks[nextInstructionIndex]);
            }
        }
    }

    private void ConstructBasicBlocks()
    {
        var instructions = _disassembledMsilMethod.Instructions;
        var currentIndex = 0;

        foreach (var (startOffset, endOffset) in _basicBlockMapping.SkipLast(1))
        {
            currentIndex++;
            var basicBlockInstructions = new List<MsilInstruction<IMsilOperand>>();
            for (var i = _instructionOffsetMapping[startOffset]; i <= _instructionOffsetMapping[endOffset] - instructions[_instructionOffsetMapping[endOffset]].Size; i++)
            {
                basicBlockInstructions.Add(instructions[i]);
            }

            if (basicBlockInstructions.Count == 0)
            {
                basicBlockInstructions.Add(instructions[_instructionOffsetMapping[startOffset]]);
            }

            _basicBlocks.Add(startOffset, new BasicBlock($"block_{currentIndex}", basicBlockInstructions));
        }

        var lastBlockInstructions = new List<MsilInstruction<IMsilOperand>>();
        for (var i = _instructionOffsetMapping[_basicBlockMapping.Last().Key]; i < instructions.Length; i++)
        {
            lastBlockInstructions.Add(instructions[i]);
        }

        _basicBlocks.Add(_basicBlockMapping.Last().Key, new BasicBlock($"block_{_basicBlocks.Count + 1}", lastBlockInstructions));
    }

    private void FindBasicBlockMapping()
    {
        var instructions = _disassembledMsilMethod.Instructions;
        var leaders = new HashSet<int> { 0 };

        foreach (var instruction in instructions)
        {
            if (instruction.IsTerminator)
            {
                _ = leaders.Add(instruction.Offset + instruction.Size);
            }

            if (instruction.IsBranch)
            {
                foreach (var target in instruction.GetBranchTargets())
                {
                    _ = leaders.Add(target);
                }
            }
        }

        var sortedLeaders = leaders.OrderBy(x => x).ToArray();

        for (var i = 0; i < sortedLeaders.Length - 1; i++)
        {
            _basicBlockMapping.Add(sortedLeaders[i], sortedLeaders[i + 1]);
        }
    }

    private void CreateInstructionOffsetMapping()
    {
        var instructions = _disassembledMsilMethod.Instructions;
        for (var i = 0; i < instructions.Length; i++)
        {
            _instructionOffsetMapping.Add(instructions[i].Offset, i);
        }
    }
}