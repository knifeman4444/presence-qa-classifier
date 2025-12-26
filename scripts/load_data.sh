#!/bin/bash

# Script to download and prepare data for presence-qa-classifier

set -e

# Create data directories if they don't exist
mkdir -p data/boolq
mkdir -p data/multinli_1.0

echo "Downloading MultiNLI dataset..."
# Download MultiNLI zip file
if [ ! -f "data/multinli_1.0.zip" ]; then
    wget -O data/multinli_1.0.zip https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
fi

# Extract MultiNLI if directory is empty or doesn't exist
if [ ! -d "data/multinli_1.0" ] || [ -z "$(ls -A data/multinli_1.0)" ]; then
    echo "Extracting MultiNLI dataset..."
    unzip -q data/multinli_1.0.zip -d data/
    # Clean up zip file after extraction
    rm data/multinli_1.0.zip
fi

echo "Downloading BoolQ dataset..."
# Download BoolQ train file
echo "Downloading BoolQ train file..."
wget -O data/boolq/boolq_3l_train_full.jsonl https://raw.githubusercontent.com/CogComp/Yes-No-or-IDK/refs/heads/master/DATA/BoolQ_3L/train_full.json

# Download BoolQ dev file and rename to test
echo "Downloading BoolQ dev file (as test)..."
wget -O data/boolq/boolq_3l_test_full.jsonl https://raw.githubusercontent.com/CogComp/Yes-No-or-IDK/refs/heads/master/DATA/BoolQ_3L/dev_full.json

echo "Data loading complete!"
echo "MultiNLI data: data/multinli_1.0/"
echo "BoolQ data: data/boolq/"
