#!/bin/bash

TARGET_DIR=$(pwd)

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

if ! command -v kaggle &> /dev/null
then
    echo "Kaggle CLI n'est pas installé. Installation en cours..."
    pip install kaggle --upgrade
fi

if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Fichier kaggle.json non trouvé dans ~/.kaggle. Veuillez le placer et réessayer."
    exit 1
fi

echo "Téléchargement du dataset Skin Cancer MNIST: HAM10000 dans $TARGET_DIR"
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p "$TARGET_DIR" --unzip

echo "Téléchargement terminé."