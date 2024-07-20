// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.IR;

namespace Sci.NET.Accelerators.Rewriter;

internal class CfgBuilder
{
    private readonly DisassembledMsilMethod _disassembledMethod;
    private readonly HashSet<int> _leaders;
    private readonly HashSet<int> _terminators;
    private readonly Dictionary<int, BasicBlock> _basicBlocks;
    private readonly Dictionary<int, List<BasicBlock>> _successorMapping;

    public CfgBuilder(DisassembledMsilMethod disassembledMsilMethod)
    {
        _disassembledMethod = disassembledMsilMethod;
        _successorMapping = new Dictionary<int, List<BasicBlock>>();
        _leaders = new HashSet<int>();
        _terminators = new HashSet<int>();
        _basicBlocks = new Dictionary<int, BasicBlock>();
    }

    public List<BasicBlock> Build()
    {
        FindLeadersTerminators();
        ConstructBasicBlocks();
        ComputeSuccessors();
        AddPredecessorsAndSuccessors();

        return _basicBlocks.Values.ToList();
    }

    private void ConstructBasicBlocks()
    {
        var sortedLeaders = _leaders.OrderBy(x => x).ToArray();
        var sortedTerminators = _terminators.OrderBy(x => x).ToArray();

        for (var i = 0; i < sortedLeaders.Length; i++)
        {
            var leader = sortedLeaders[i];
            var terminator = sortedTerminators[i];

            var instructions = new List<MsilInstruction<IMsilOperand>>();
            for (var j = leader; j < terminator; j++)
            {
                instructions.Add(_disassembledMethod.Instructions[j]);
            }

            _basicBlocks.Add(i, new BasicBlock($"block_{i}", instructions));
        }
    }

    private void FindLeadersTerminators()
    {
        var leaders = new HashSet<int> { 0 };
        var terminators = new HashSet<int>();

        foreach (var instruction in _disassembledMethod.Instructions)
        {
            if (instruction.IsBranch)
            {
                foreach (var target in instruction.GetBranchTargetInstructionIndices(_disassembledMethod))
                {
                    _ = leaders.Add(target);
                }
            }
        }

        _leaders.UnionWith(leaders.OrderBy(x => x));
        _terminators.UnionWith(_leaders.Skip(1).Select(x => x - 1).Append(_disassembledMethod.Instructions.Length - 1));
    }

    private void ComputeSuccessors()
    {
        foreach (var basicBlock in _basicBlocks.SkipLast(1))
        {
            var lastInstruction = basicBlock.Value.MsilInstructions.Last();

            if (!_successorMapping.TryGetValue(basicBlock.Key, out List<BasicBlock>? value))
            {
                value = new List<BasicBlock>();
                _successorMapping.Add(basicBlock.Key, value);
            }

            if (lastInstruction.IsBranch)
            {
                foreach (var target in lastInstruction.GetBranchTargetInstructions(_disassembledMethod))
                {
                    var targetBlocks = _basicBlocks.Where(x => x.Value.MsilInstructions.Contains(target)).ToList();
                    value.AddRange(targetBlocks.Select(x => x.Value));
                }
            }
            else
            {
                if (!_successorMapping.ContainsKey(basicBlock.Key + 1))
                {
                    _successorMapping.Add(basicBlock.Key + 1, new List<BasicBlock>());
                }

                value.Add(_basicBlocks[basicBlock.Key + 1]);
            }
        }
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
}