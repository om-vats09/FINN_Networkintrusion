source bifrom qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import MoveScalarLinearPastInvariants
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferQuantizedMatrixVectorActivation,
    InferThresholdingLayer,
)
import os

os.makedirs('models/finn_hw', exist_ok=True)

print("Loading clean QONNX model...")
model = ModelWrapper("models/nids_4bit_qonnx_clean.onnx")

print("\nConverting QONNX to FINN internal format...")
model = model.transform(ConvertQONNXtoFINN())
print("After ConvertQONNXtoFINN:")
for i, node in enumerate(model.graph.node):
    print(f"  [{i:02d}] {node.op_type}")

print("\nStreamlining...")
for _ in range(3):
    model = model.transform(Streamline())
    model = model.transform(MoveScalarLinearPastInvariants())
model = model.transform(InferDataTypes())
model = model.transform(GiveUniqueNodeNames())
model.save("models/finn_hw/nids_streamlined_v3.onnx")

print("\nAfter streamlining:")
for i, node in enumerate(model.graph.node):
    print(f"  [{i:02d}] {node.op_type}")

print("\nConverting to HW layers...")
model = model.transform(InferQuantizedMatrixVectorActivation())
model = model.transform(InferThresholdingLayer())
model = model.transform(GiveUniqueNodeNames())
model.save("models/finn_hw/nids_hw_v3.onnx")

print("\nFinal HW graph:")
for i, node in enumerate(model.graph.node):
    print(f"  [{i:02d}] {node.op_type}")

print("\nDone.")
