import numpy as np
from collections import defaultdict
import json

dnns = {"darknet": range(0, 20 + 1), "resnet": range(0, 20 + 1), "squeezenet": range(0, 20 + 1)}
baseline = "resnet", 0
fps = defaultdict(list)
for net_name, steps in dnns.items():
    for step in steps:
        filename = f"cropfollow/data/{net_name}_{step}.txt"
        with open(filename, "r") as f:
            lines = f.readlines()[4:]
            data = [[float(s) for s in line.strip().split(",")] for line in lines]
        data = np.array(data)
        # Skip the first sample (warmup)
        t_dnn, t_overall = data[1:].T
        t0_dnn, t0_overall = t_dnn.mean(), t_overall.mean()
        # Calculate the time for postprocessing
        fps[net_name].append((1 / t0_dnn, 1 / t0_overall))
with open("cropfollow/autotuner_data/fps_l1.json") as f:
    data = json.load(f)
for k, vs in data.items():
    for d, (dnn_fps, overall_fps) in zip(vs, fps[k]):
        d["fps_dnn"] = dnn_fps
        d["fps_node"] = overall_fps
with open("cropfollow/autotuner_data/fps_l1.json", "w") as f:
    json.dump(data, f, indent=2)

dnns = {"darknet": range(0, 19 + 1), "vgg": range(0, 19 + 1)}
baseline = "vgg", 0
timed = defaultdict(list)
# Non-DNN overhead
t_overhead = []
for net_name, steps in dnns.items():
    for step in steps:
        filename = f"gem/data/{net_name}_{step}.txt"
        with open(filename, "r") as f:
            lines = f.readlines()[1:]
            data = [[float(s) for s in line.strip().split(",")] for line in lines]
        data = np.array(data)
        # Skip the first sample (warmup)
        t_dnn, t_overall = data[1:].T
        t0_dnn = t_dnn.mean()
        # Calculate the time for postprocessing
        t_overhead.append(t_overall[t_overall != 0] - t0_dnn)
        timed[net_name].append(t0_dnn)
t_overhead = np.concatenate(t_overhead)
t0_overhead = t_overhead.mean()
print(f"Overhead: {t0_overhead:.4f} +- {t_overhead.std():.4f}")
timed = {k: np.array(vs) for k, vs in timed.items()}
baseline_time = timed[baseline[0]][baseline[1]]

data = {}
for k, arr in timed.items():
    data[k] = [{"fps_dnn": 1 / v, "fps_none": 1 / (v + t0_overhead)} for v in arr]
json.dump(data, open("gem/results/fps_acc.json", "w"), indent=2)
