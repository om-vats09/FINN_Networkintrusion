from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    FactorOutMulSignMagnitude,
    AbsorbSignBiasIntoMultiThreshold,
)
from finn.transformation.streamline.reorder import MoveScalarLinearPastInvariants
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferQuantizedMatrixVectorActivation,
    InferThresholdingLayer,
    InferBinaryMatrixVectorActivation,
)
import os

os.makedirs('models/finn_hw', exist_ok=True)

print("Loading QONNX model...")
model = ModelWrapper("models/nids_4bit_qonnx.onnx")
print(f"  Nodes: {len(model.graph.node)}")

print("\nRunning full streamlining pipeline...")
for i in range(3):
    model = model.transform(Streamline())
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(Streamline())

model = model.transform(InferShapes())
model = model.transform(InferDataTypes())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model.save("models/finn_hw/nids_streamlined_full.onnx")

print("\nAfter full streamlining:")
for i, node in enumerate(model.graph.node):
    print(f"  [{i:02d}] {node.op_type:40s}")

print("\nConverting to HW layers...")
model = model.transform(InferQuantizedMatrixVectorActivation())
model = model.transform(InferBinaryMatrixVectorActivation())
model = model.transform(InferThresholdingLayer())
model = model.transform(GiveUniqueNodeNames())
model.save("models/finn_hw/nids_hw_layers.onnx")

print("\nFinal HW graph:")
for i, node in enumerate(model.graph.node):
    print(f"  [{i:02d}] {node.op_type:40s}")

print("\nDone.")
