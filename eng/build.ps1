    [CmdletBinding(PositionalBinding = $false)]
    Param(
        [string][Alias('c')]$Configuration = "Debug",
        [string]$DotnetInstallDir = "$PSScriptRoot\dotnet",
        [string]$EngineeringRoot = "$PSScriptRoot",
        [string]$RepoRoot = "$PSScriptRoot\..",
        [string]$CudaVersion = "12.3.1",
        [string]$PackageVersion = "0.0.0",
        [bool]$CleanBuild = $false,
        [bool]$RunTests = $true,
        [bool]$BuildNugetPackages = $true
    )

    $GlobalJsonFile = Resolve-Path "$RepoRoot\global.json"
    $Dotnet = Join-Path $DotnetInstallDir "dotnet.exe"

    # Install dotnet if not installed
    $DotnetInstallScript = Join-Path $EngineeringRoot "dotnet-install.ps1"
    
    # Install dotnet
    &$DotnetInstallScript -InstallDir $DotnetInstallDir -JSonFile $GlobalJsonFile -Architecture x64 -Verbose

    # Set Repo Root
    $SolutionPath = Resolve-Path $RepoRoot\Sci.NET.sln

    # Install dotnet 
    &$Dotnet tool update -g docfx

    if ($CleanBuild)
    {
        &$Dotnet clean
    }

    # Restore
    &$Dotnet restore $SolutionPath -s https://api.nuget.org/v3/index.json
    
    # Build
    &$Dotnet build $SolutionPath -c $Configuration

    if ($RunTests)
    {
        # Test
        &$Dotnet test $SolutionPath -c $Configuration   
    }    
    
    if ($BuildNugetPackages)
    {
        &$Dotnet pack $SolutionPath -c Release /p:Version=$PackageVersion
    }
