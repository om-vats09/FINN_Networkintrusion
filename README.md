the link of the colab notebook is :https://colab.research.google.com/drive/14ASwORoM70xddq-ZnWJnG_iYuC234jVb?usp=sharing
and the diffrent aspect I have explained in the part as followed into it 



Always run this on the codespaces for it:
source env/bin/activate 
python3 preprocess.py 
python3 train.py 
python3 export_onnx.py 
python3 finn_pipeline.py







cd /workspaces/FINN_Networkintrusion
python3 -m venv env
source env/bin/activate
pip install torch torchvision brevitas onnx onnxruntime scikit-learn pandas numpy matplotlib seaborn qonnx onnxscript setuptools==69.5.1
python3 preprocess.py
python3 train.py
