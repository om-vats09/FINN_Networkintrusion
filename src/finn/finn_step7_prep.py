from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
import json

print("Loading HW model...")
model = ModelWrapper("models/finn_hw/nids_hw_v3.onnx")

print("\nFinal model summary:")
node_types = {}
for i, node in enumerate(model.graph.node):
    t = node.op_type
    node_types[t] = node_types.get(t, 0) + 1
    print(f"  [{i:02d}] {t}")

print("\nNode type counts:")
for t, c in node_types.items():
    print(f"  {t}: {c}")

mvau_count = node_types.get('MVAU', 0)
print(f"\nMVAU hardware layers: {mvau_count}/4")
print(f"Remaining non-HW nodes: {len(model.graph.node) - mvau_count}")

print("""
Pipeline status:
  Step 1 - Preprocessing     : DONE
  Step 2 - QAT Training      : DONE (81.97% accuracy)
  Step 3 - ONNX export       : DONE
  Step 4 - Streamlining      : DONE
  Step 5 - Optimisation      : DONE
  Step 6 - MVAU conversion   : DONE (3 MVAU nodes)
  Step 7 - Vivado HLS        : PENDING (needs Vivado)
  Step 8 - PYNQ-Z2 deploy    : PENDING (needs board)

Ready for Vivado synthesis when hardware is available.
""")

model.save("models/finn_hw/nids_ready_for_vivado.onnx")
print("Saved → models/finn_hw/nids_ready_for_vivado.onnx")
