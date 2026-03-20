#!/bin/bash
echo "Setting up IDS FINN environment..."

python3 -m venv env
source env/bin/activate

pip install --quiet torch torchvision brevitas onnx onnxruntime \
    scikit-learn pandas numpy matplotlib seaborn \
    qonnx onnxscript setuptools==69.5.1

echo "Regenerating data and models..."
python3 preprocess.py
python3 train.py
python3 export_onnx.py
python3 finn_pipeline.py

echo "Done. Environment ready."