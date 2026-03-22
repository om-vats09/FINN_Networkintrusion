from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import MoveScalarLinearPastInvariants
from finn.transformation.streamline.collapse_repeated import CollapseRepeatedMul
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
import os

os.makedirs('models/finn_hw', exist_ok=True)

print("Loading QONNX model...")
model = ModelWrapper("models/nids_4bit_qonnx.onnx")
print(f"  Nodes: {len(model.graph.node)}")

print("\nStep 4 — Streamlining...")
model = model.transform(Streamline())
model.save("models/finn_hw/nids_streamlined.onnx")
print("  Saved → models/finn_hw/nids_streamlined.onnx")

print("\nStep 5 — Reorder and collapse...")
model = model.transform(MoveScalarLinearPastInvariants())
model = model.transform(CollapseRepeatedMul())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model.save("models/finn_hw/nids_optimised.onnx")
print("  Saved → models/finn_hw/nids_optimised.onnx")

print("\nGraph after streamlining:")
for i, node in enumerate(model.graph.node):
    print(f"  [{i:02d}] {node.op_type:40s}")

print("\nDone.")
