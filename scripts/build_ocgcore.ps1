Write-Host "Clonage du dépôt ocgcore..."
git clone --recursive https://github.com/knight00/ocgcore-KCG.git core/ocgcore_src

Set-Location core/ocgcore_src

Write-Host "Génération des Makefiles (vs2022 par défaut pour Windows)..."
# Note: Sur Windows, on génère généralement pour Visual Studio.
# Si vous utilisez MinGW/MSYS2, utilisez 'premake5 gmake' et 'mingw32-make'
premake5 vs2022

Write-Host "Compilation en cours (via MSBuild)..."
# Assurez-vous que msbuild est dans le PATH
msbuild ocgcore.sln /p:Configuration=Release /p:Platform=x64

Write-Host "Déplacement du binaire compilé..."
if (Test-Path "bin\Release\ocgcore.dll") {
    Copy-Item "bin\Release\ocgcore.dll" "..\ygoenv\" -Force
    Write-Host "Compilation terminée avec succès ! ocgcore.dll est dans core/ygoenv/"
} else {
    Write-Host "Attention: Le binaire n'a pas été trouvé. Vérifiez les logs de compilation." -ForegroundColor Red
}

Set-Location ..\..
