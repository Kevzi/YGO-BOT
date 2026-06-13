#!/bin/bash
echo "Clonage du dépôt ocgcore..."
if [ ! -d "core/ocgcore_src" ]; then
    git clone --recursive https://github.com/knight00/ocgcore-KCG.git core/ocgcore_src
fi

cd core/ocgcore_src || exit 1

echo "Vérification de premake5..."
if ! command -v premake5 &> /dev/null; then
    echo "Téléchargement de premake5..."
    wget -q https://github.com/premake/premake-core/releases/download/v5.0.0-alpha16/premake-5.0.0-alpha16-linux.tar.gz
    tar -xzf premake-5.0.0-alpha16-linux.tar.gz
    chmod +x premake5
    export PATH=$PATH:$(pwd)
fi

echo "Génération des Makefiles..."
premake5 gmake || ./premake5 gmake || { echo "Échec de premake5."; exit 1; }

echo "Compilation en cours..."
make -C build config=release ocgcoreshared

echo "Déplacement du binaire compilé..."
cp bin/release/libocgcore.so ../ygoenv/ || echo "Erreur de copie"
echo "Compilation terminée avec succès !"
