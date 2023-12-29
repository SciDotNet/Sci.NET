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

    # Install dotnet 
    &$Dotnet tool update -g docfx

    if ($CleanBuild)
    {
        &$Dotnet clean
    }

    # Build CUDA Download Tool
    &$Dotnet build $EngineeringRoot\Sci.NET.CudaDownloaderTool\Sci.NET.CudaDownloaderTool.sln
    &$EngineeringRoot\Sci.NET.CudaDownloaderTool\Sci.NET.CUDA.Downloader\bin\Debug\net8.0\Sci.NET.CUDA.Downloader.exe 12.3.1 $EngineeringRoot\CUDA\
    
    # Restore
    &$Dotnet restore $RepoRoot\Sci.NET.sln -s https://api.nuget.org/v3/index.json
    
    # Build
    &$Dotnet build $RepoRoot\Sci.NET.sln -c $Configuration

    if ($RunTests)
    {
        # Test
        &$Dotnet test $RepoRoot\Sci.NET.sln -c $Configuration   
    }    
    
    if ($BuildNugetPackages)
    {
        # Build Nuget packages
        &$Dotnet pack $RepoRoot\Sci.NET.sln -o $RepoRoot\artifacts\nuget -c $Configuration -p:PackageVersion = $PackageVersion   
    }