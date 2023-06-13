from pathlib import Path

import numpy as np

from approxcaliper.infer import CallableModel


def parse_args():
    from argparse import ArgumentParser

    def parse_tuple(s: str):
        return tuple([int(ss) for ss in s.split(",")])

    parser = ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("input_shape", type=parse_tuple)
    return parser.parse_args()


def timereps(model: CallableModel, input: np.ndarray, n_repeats: int):
    from time import time

    start = time()
    for _ in range(0, n_repeats):
        model(input)
    end = time()
    return (end - start) / n_repeats


if __name__ == "__main__":
    args = parse_args()
    model = CallableModel.from_onnx(args.model, "GPU")
    image_np_4d = np.zeros(args.input_shape)
    print(timereps(model, image_np_4d, 100))
