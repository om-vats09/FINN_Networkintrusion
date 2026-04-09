from pathlib import Path
import json

import numpy as np
import torch
from brevitas.export import export_qonnx
from model import build_model
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes


FINN_DIR = Path("models/finn")
FINN_HW_DIR = Path("models/finn_hw")
STATE_DICT = Path("models/model_4bit.pt")
TEST_DATA = Path("data/X_test.npy")


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_finn():
    try:
        from finn.transformation.fpgadataflow.convert_to_hw_layers import (
            InferQuantizedMatrixVectorActivation,
            InferThresholdingLayer,
        )
        from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
        from finn.transformation.streamline import Streamline
        from finn.transformation.streamline.reorder import MoveScalarLinearPastInvariants
    except ModuleNotFoundError as exc:
        if exc.name == "finn":
            raise SystemExit(
                "\nFINN is not installed in this Python environment.\n"
                "This project can train/export locally, but FINN conversion/build steps must run in a FINN environment.\n"
                "Official FINN docs recommend the Docker-based setup:\n"
                "https://finn.readthedocs.io/en/latest/getting_started.html\n\n"
                "What to do next:\n"
                "1. Keep using this repo for preprocess.py and train.py.\n"
                "2. Move the repo into a FINN Docker or host environment with Vivado/Vitis available.\n"
                "3. Run python3 finn_prepare_4bit_deploy.py there.\n"
                "4. Then run python3 finn_build_accelerator.py --board zedboard.\n"
            ) from exc
        raise

    return (
        InferQuantizedMatrixVectorActivation,
        InferThresholdingLayer,
        ConvertQONNXtoFINN,
        Streamline,
        MoveScalarLinearPastInvariants,
    )


def export_clean_qonnx(model: torch.nn.Module, sample: np.ndarray) -> Path:
    export_path = FINN_DIR / "nids_4bit_qonnx.onnx"
    clean_path = FINN_DIR / "nids_4bit_qonnx_clean.onnx"

    print("Exporting 4-bit model to QONNX...")
    export_qonnx(model, input_t=torch.tensor(sample), export_path=str(export_path), opset_version=11)

    print("Cleaning exported graph...")
    qonnx_model = ModelWrapper(str(export_path))
    qonnx_model = qonnx_model.transform(InferShapes())
    qonnx_model = qonnx_model.transform(FoldConstants())
    qonnx_model = qonnx_model.transform(GiveUniqueNodeNames())
    qonnx_model = qonnx_model.transform(GiveReadableTensorNames())
    qonnx_model = qonnx_model.transform(RemoveStaticGraphInputs())
    qonnx_model.save(str(clean_path))
    print(f"Saved clean QONNX model -> {clean_path}")
    return clean_path


def convert_to_finn_hw(clean_path: Path) -> dict:
    (
        InferQuantizedMatrixVectorActivation,
        InferThresholdingLayer,
        ConvertQONNXtoFINN,
        Streamline,
        MoveScalarLinearPastInvariants,
    ) = require_finn()

    streamlined_path = FINN_HW_DIR / "nids_4bit_streamlined.onnx"
    ready_path = FINN_HW_DIR / "nids_4bit_ready.onnx"

    print("Converting clean QONNX model into FINN format...")
    model = ModelWrapper(str(clean_path))
    model = model.transform(ConvertQONNXtoFINN())

    print("Running FINN streamlining passes...")
    for _ in range(3):
        model = model.transform(Streamline())
        model = model.transform(MoveScalarLinearPastInvariants())

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model.save(str(streamlined_path))
    print(f"Saved streamlined FINN model -> {streamlined_path}")

    print("Inferring hardware layers...")
    model = model.transform(InferQuantizedMatrixVectorActivation())
    model = model.transform(InferThresholdingLayer())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model.save(str(ready_path))
    print(f"Saved hardware-ready FINN model -> {ready_path}")

    node_types = {}
    for node in model.graph.node:
        node_types[node.op_type] = node_types.get(node.op_type, 0) + 1
    return {
        "streamlined_model": str(streamlined_path),
        "hardware_ready_model": str(ready_path),
        "node_types": node_types,
    }


def save_verification_data(model: torch.nn.Module, sample: np.ndarray) -> dict:
    input_path = FINN_HW_DIR / "input.npy"
    expected_path = FINN_HW_DIR / "expected_output.npy"

    print("Saving one-sample verification vectors...")
    with torch.no_grad():
        expected = model(torch.tensor(sample)).cpu().numpy().astype(np.float32)

    np.save(input_path, sample.astype(np.float32))
    np.save(expected_path, expected)
    print(f"Saved verification input -> {input_path}")
    print(f"Saved verification output -> {expected_path}")
    return {
        "verification_input": str(input_path),
        "verification_output": str(expected_path),
    }


def main() -> None:
    FINN_DIR.mkdir(parents=True, exist_ok=True)
    FINN_HW_DIR.mkdir(parents=True, exist_ok=True)

    ensure_exists(STATE_DICT)
    ensure_exists(TEST_DATA)

    print("=" * 60)
    print("Preparing 4-bit FINN deployment artifacts")
    print("=" * 60)

    print("Loading trained 4-bit model...")
    model = build_model(4)
    model.load_state_dict(torch.load(STATE_DICT, map_location="cpu"))
    model.eval()

    sample = np.load(TEST_DATA)[:1].astype(np.float32)

    clean_path = export_clean_qonnx(model, sample)
    hw_summary = convert_to_finn_hw(clean_path)
    verification_summary = save_verification_data(model, sample)

    manifest = {
        "bit_width": 4,
        "deploy_target": "zedboard",
        "model_state_dict": str(STATE_DICT),
        "qonnx_clean_model": str(clean_path),
        **hw_summary,
        **verification_summary,
    }

    manifest_path = FINN_HW_DIR / "deploy_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved deployment manifest -> {manifest_path}")

    print("\nNode type summary:")
    for node_type, count in sorted(hw_summary["node_types"].items()):
        print(f"  {node_type}: {count}")

    print("\nNext step:")
    print("  python3 finn_build_accelerator.py --board zedboard")


if __name__ == "__main__":
    main()
