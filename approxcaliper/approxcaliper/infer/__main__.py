import argparse
from pathlib import Path

from approxcaliper.infer.models import compile_onnx_to_trt


def scan_file_or_folder(input_path: Path, pattern: str):
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        return sorted(input_path.rglob(pattern))
    else:
        raise RuntimeError(f"Path {input_path} not found")


def main():
    parser = argparse.ArgumentParser(
        description="Convert onnx model to have fixed batchsize"
    )
    parser.add_argument(
        "-b", "--batchsize", type=int, default=1, help="batchsize to use (default 1)"
    )
    parser.add_argument(
        "input", type=Path, help="input (onnx file or folder with onnx file)"
    )
    args = parser.parse_args()
    for in_path in scan_file_or_folder(args.input, "*.onnx"):
        out_path = in_path.with_suffix(".pkl")
        if not out_path.is_file():
            print(f"{in_path} -> {out_path}")
            compile_onnx_to_trt(args.batchsize, in_path, out_path)
        else:
            print(f"Skipping {in_path} as {out_path} exists")


if __name__ == "__main__":
    main()
