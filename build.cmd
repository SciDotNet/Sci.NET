dotnet tool install --global Microsoft.VisualStudio.SlnGen.Tool
cd eng
powershell.exe -noexit "& '.\build.ps1 ' -C Release"