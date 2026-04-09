import argparse
from pathlib import Path

MODEL_PATH = Path("models/finn_hw/nids_4bit_ready.onnx")
VERIFY_INPUT = Path("models/finn_hw/input.npy")
VERIFY_OUTPUT = Path("models/finn_hw/expected_output.npy")
ZEDBOARD_FPGA_PART = "xc7z020clg484-1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FINN board build for this project.")
    parser.add_argument(
        "--board",
        choices=["zedboard", "pynq-z2"],
        default="zedboard",
        help="Build mode. ZedBoard uses stitched IP for manual Vivado integration.",
    )
    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Hardware-ready FINN model produced by finn_prepare_4bit_deploy.py",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where FINN build products will be written.",
    )
    parser.add_argument(
        "--clock-period-ns",
        type=float,
        default=10.0,
        help="Target synthesis clock period in ns. 10.0 ns corresponds to 100 MHz.",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=None,
        help="Optional performance target for FINN folding.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose FINN build logging.",
    )
    return parser.parse_args()


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_finn_builder():
    try:
        from finn.builder.build_dataflow import build_dataflow_cfg
        from finn.builder.build_dataflow_config import (
            DataflowBuildConfig,
            DataflowOutputType,
            ShellFlowType,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "finn":
            raise SystemExit(
                "\nFINN is not installed in this Python environment.\n"
                "The accelerator build must run inside a FINN environment with the supported toolchain.\n"
                "Official setup instructions:\n"
                "https://finn.readthedocs.io/en/latest/getting_started.html\n"
            ) from exc
        raise

    return build_dataflow_cfg, DataflowBuildConfig, DataflowOutputType, ShellFlowType


def get_build_config(args: argparse.Namespace):
    _, DataflowBuildConfig, DataflowOutputType, ShellFlowType = require_finn_builder()
    output_dir = Path(args.output_dir or f"build/{args.board}")
    output_dir.mkdir(parents=True, exist_ok=True)

    common_kwargs = {
        "output_dir": str(output_dir),
        "synth_clk_period_ns": args.clock_period_ns,
        "verify_input_npy": str(VERIFY_INPUT),
        "verify_expected_output_npy": str(VERIFY_OUTPUT),
        "target_fps": args.target_fps,
        "save_intermediate_models": True,
        "verbose": args.verbose,
    }

    if args.board == "zedboard":
        return DataflowBuildConfig(
            generate_outputs=[
                DataflowOutputType.ESTIMATE_REPORTS,
                DataflowOutputType.STITCHED_IP,
                DataflowOutputType.OOC_SYNTH,
            ],
            fpga_part=ZEDBOARD_FPGA_PART,
            **common_kwargs,
        )

    return DataflowBuildConfig(
        generate_outputs=[
            DataflowOutputType.ESTIMATE_REPORTS,
            DataflowOutputType.BITFILE,
            DataflowOutputType.PYNQ_DRIVER,
            DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
        board="Pynq-Z2",
        shell_flow_type=ShellFlowType.VIVADO_ZYNQ,
        **common_kwargs,
    )


def main() -> None:
    build_dataflow_cfg, _, _, _ = require_finn_builder()
    args = parse_args()
    model_path = Path(args.model)

    ensure_exists(model_path)
    ensure_exists(VERIFY_INPUT)
    ensure_exists(VERIFY_OUTPUT)

    print("=" * 60)
    print(f"Launching FINN build for {args.board}")
    print("=" * 60)
    print(f"Model          : {model_path}")
    print(f"Clock period   : {args.clock_period_ns} ns")
    if args.target_fps is not None:
        print(f"Target FPS     : {args.target_fps}")

    if args.board == "zedboard":
        print("\nBuild mode summary:")
        print("  ZedBoard is treated as a custom Zynq target.")
        print("  FINN will generate stitched IP and reports.")
        print("  You will finish block design, bitstream generation, and software handoff in Vivado/Vitis.")
    else:
        print("\nBuild mode summary:")
        print("  Pynq-Z2 uses FINN's supported Vivado Zynq shell flow.")
        print("  FINN will attempt full bitfile, driver, and deployment package generation.")

    build_cfg = get_build_config(args)
    build_dataflow_cfg(str(model_path), build_cfg)

    print("\nFINN build completed.")
    print(f"Artifacts written under: {build_cfg.output_dir}")


if __name__ == "__main__":
    main()
