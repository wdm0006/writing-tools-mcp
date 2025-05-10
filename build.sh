#!/bin/bash

set -e

rm -rf build

# 1. Setup virtual environment if not already present
if [ ! -d ".buildvenv" ]; then
    uv venv .buildvenv --seed
fi

source .buildvenv/bin/activate

# 2. Install dependencies
uv pip install --upgrade pip
uv pip install -e . 
uv pip install nuitka 

# 3. Download spaCy English model if not present
python -m spacy download en_core_web_sm

# 4. Compile with Nuitka
python -m nuitka \
    --follow-imports \
    --enable-plugin=pyside6 \
    --enable-plugin=spacy \
    --spacy-language-model=en_core_web_sm \
    --standalone \
    --macos-disable-console \
    --output-dir=build \
    --include-data-files=$(python -c "import pyphen, os; print(os.path.dirname(pyphen.__file__))")/dictionaries/*=pyphen/dictionaries/ \
    --include-data-files=$(python -c "import spellchecker, os; print(os.path.dirname(spellchecker.__file__))")/resources/*=spellchecker/resources/ \
    --include-data-dir=$(python -c "import site; print(site.getsitepackages()[0])")/PySide6/Qt=PySide6/Qt \
    app/main.py


    # --macos-create-app-bundle \
    # --onefile \
    # --macos-app-name="WritingToolsMCP" \
    # --macos-app-version=1.0.0 \
    # --macos-app-icon=images/icon.icns \

echo "Packaging complete! Run your app with:"
echo "open ./build/main.app"