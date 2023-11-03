    [CmdletBinding(PositionalBinding = $false)]
    Param(
        [string][Alias('c')]$Configuration = "Debug",
        [string]$CMakeBuildDirectory = "$PSScriptRoot\cmake_build",
        [string]$DotnetInstallDir = "$PSScriptRoot\dotnet",
        [string]$EngineeringRoot = "$PSScriptRoot",
        [string]$RepoRoot = "$PSScriptRoot\..",
        [string]$PackageVersion = "0.0.0",
        [bool]$CleanBuild = $false,
        [bool]$BuildNative = $true,
        [bool]$RunTests = $true,
        [bool]$BuildNugetPackages = $true
    )

    $GlobalJsonFile = Resolve-Path "$RepoRoot\global.json"
    $Dotnet = Join-Path $DotnetInstallDir "dotnet.exe"

    if ($CleanBuild)
    {
        if ($CleanBuild)
        {
            Write-Host "Cleaning build directory"
            Remove-Item -Recurse -Force $CMakeBuildDirectory
        }

        if (-Not(Test-Path $CMakeBuildDirectory))
        {
            Write-Host "Creating build directory"
            New-Item -ItemType Directory -Force -Path $CMakeBuildDirectory
        }

    }

    # Install dotnet if not installed
    $DotnetInstallScript = Join-Path $EngineeringRoot "dotnet-install.ps1"
    
    # Change when .NET 8 is in GA
    # &$DotnetInstallScript -InstallDir $DotnetInstallDir -JSonFile $GlobalJsonFile -Architecture x64 -Quality preview -Verbose
    &$DotnetInstallScript -Channel 8.0.1xx -InstallDir ./dotnet -Verbose

    # Build cmake
    cmake -S $RepoRoot -B $CMakeBuildDirectory

    # Build the native code
    $vswhere = Join-Path $EngineeringRoot "vswhere.exe"
    $buildProject = Resolve-Path (Join-Path $CMakeBuildDirectory "Sci.NET.Native.sln")
    $msbuildPath = (&$vswhere -latest -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe)
    &$msbuildPath "`"$buildProject`"" /property:Configuration=Release

    # Build
    &$Dotnet restore $RepoRoot\Sci.NET.sln -s https://api.nuget.org/v3/index.json
    &$Dotnet build $RepoRoot\Sci.NET.sln -c Release
    
    # Test
    &$Dotnet test $RepoRoot\Sci.NET.sln -c Test
    
    # Build Nuget packages
    &$Dotnet pack $RepoRoot\Sci.NET.sln -o $RepoRoot\artifacts\nuget -c Release -p:PackageVersion = $PackageVersion   