[CmdletBinding(PositionalBinding = $false)]
    Param(
        [string][Alias('c')]$Configuration = "Debug",
        [string]$CMakeBuildDirectory = "$PSScriptRoot\cmake_build",
        [string]$DotnetInstallDir = "$PSScriptRoot\dotnet",
        [string]$EngineeringRoot = "$PSScriptRoot",
        [string]$RepoRoot = "$PSScriptRoot\..",
        [switch]$BuildCMake = $true,
        [switch]$BuildSolution = $false,
        [switch]$TestSolution = $true
    )

    $RepoRoot = Resolve-Path($RepoRoot)
    $GlobalJsonFile = Resolve-Path "$RepoRoot\global.json"
    $Dotnet = Join-Path $DotnetInstallDir "dotnet.exe"
    $vswhere = Join-Path $EngineeringRoot "vswhere.exe"
    $msbuildPath = (&$vswhere -latest -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe)
    
    # Install dotnet if not installed
    $DotnetInstallScript = Join-Path $EngineeringRoot "dotnet-install.ps1"
    &$DotnetInstallScript -InstallDir $DotnetInstallDir -JSonFile $GlobalJsonFile -Architecture x64 -Verbose

    # CMake Build
    cmake -S $RepoRoot -B $CMakeBuildDirectory

    if ($BuildCMake)
    {
        $buildProject = Resolve-Path (Join-Path $CMakeBuildDirectory "Sci.NET.Native.sln")
        &$msbuildPath $buildProject /property:Configuration=Release
    }
    
    if ($BuildSolution)
    {
        # Build
        &$Dotnet restore $RepoRoot\Sci.NET.sln
        &$Dotnet build $RepoRoot\Sci.NET.sln -c Release
    }
    
    if ($TestSolution)
    {
        &$Dotnet test $RepoRoot\Sci.NET.sln -c Debug
    }