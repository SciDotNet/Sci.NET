[CmdletBinding(PositionalBinding = $false)]
    Param(
        [string][Alias('r')]$RepoRoot = "$PSScriptRoot\..\",
        [string][Alias('m')]$MSBuildPath = ""
    )
    $SolutionName = "$RepoRoot\Sci.NET.sln"
    
    &$MSBuildPath slngen -o Sci.NET.sln --launch false $SolutionName/src/*/*.*proj $SolutionName/eng/cmake_build/Sci.NET*.*proj