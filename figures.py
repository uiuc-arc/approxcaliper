import json
import pickle as pkl
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from approxcaliper.search import Searcher
from gem.model_tuner import GEMPruneTuner
from cropfollow.model_tuner import CropFollowPruneTuner

plt.rc("font", size=16)  # controls default text sizes
plt.rc("axes", labelsize=20)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=16)  # fontsize of the tick labels
plt.rc("ytick", labelsize=16)  # fontsize of the tick labels
plt.rc("legend", fontsize=18)  # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# -----------------------------------------------------------------------------
# Before evaluation


def errorinj_example():
    fig, ax = plt.subplots()
    ax.set_xlabel("M1")
    ax.set_ylabel("M2")
    ax.plot([0, 10], [0, 10], "k-")
    ax.fill([0, 5, 5, 0], [0, 0, 5, 5], "#99dfb9")
    ax.fill([6, 10, 10, 6], [6, 6, 10, 10], "#df7f7f")
    ax.scatter([5], [5], s=15 ** 2, marker="X", color="g", zorder=2)
    ax.scatter([6], [6], s=15 ** 2, marker="X", color="r", zorder=2)

    ax.plot([0, 5, 5], [5, 5, 0], "k--")
    ax.plot([6, 6, 10], [10, 6, 6], "k--")
    ax.scatter([0, 5], [5, 0], s=15 ** 2, marker="o", color="g", zorder=2)
    ax.scatter([6, 10], [10, 6], s=15 ** 2, marker="o", color="r", zorder=2)
    ax.plot([0, 6], [5, 10], "k-")
    ax.plot([5, 10], [0, 6], "k-")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 10.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig("demo_error_inj.pdf", dpi=300)


def met_gaussian():
    from scipy.stats import norm

    diffs = np.load("cropfollow/data/model_outputs/resnet18_h_0.npy")
    mean, cov = np.mean(diffs, axis=0), np.cov(diffs, rowvar=False)
    stddev = np.sqrt(cov)
    fig, ax0 = plt.subplots()
    ax0.hist(diffs, bins=100, density=True)
    xmin, xmax = ax0.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, stddev)
    ax0.plot(x, p, label=f"$\\mathcal{{N}}({mean:.2f}, {stddev:.2f}^2)$")
    ax0.set_xlabel("Prediction error (degree)")
    ax0.set_ylabel("Frequency")
    ax0.legend()
    fig.tight_layout()
    fig.savefig("resnet18_h_gaussian.pdf")


# -----------------------------------------------------------------------------
# Reused function


def unguided(tuner_cls, tuner_file, searcher_file, output_name, **kwargs):
    searcher = Searcher.load(searcher_file)
    fig, ax = plt.subplots(figsize=(8, 4))
    searcher.plot_latest(ax, False, acc_color="#99dfb9", rej_color="#df7f7f", **kwargs)
    with open(tuner_file, "rb") as f:
        data = pkl.load(f)
        tuner_cls._plot_configs_to_ax(ax, data)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(Path(tuner_file).with_suffix(".png"), dpi=300)
    fig.savefig(output_name)


def errorinj(filepath, x_label, y_label, output_name, figsize=None, **kwargs):
    filepath = Path(filepath)
    searcher: Searcher = Searcher.load(filepath)
    fig, ax = plt.subplots(figsize=figsize)
    searcher.plot_latest(ax, acc_color="#99dfb9", rej_color="#df7f7f", **kwargs)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0)
    ax.set_ylim(0)
    fig.tight_layout()
    fig.savefig(filepath.parent / "eval_results.png", dpi=300)
    fig.savefig(output_name)


def accum_best(tunings, higher_better: bool, n_runs: int):
    niter = len(tunings)
    accum_runs = np.cumsum(np.array([int(d["empirical_run"]) for d in tunings]))
    if higher_better:
        worst = 0
        cmp = np.greater
    else:
        worst = float("inf")
        cmp = np.less
    accum_min_flops = np.array([d["perf"] if d["qos"] > 0 else worst for d in tunings])
    for i in range(1, niter):
        if cmp(accum_min_flops[i - 1], accum_min_flops[i]):
            accum_min_flops[i] = accum_min_flops[i - 1]
    ret = np.array([accum_runs, accum_min_flops]).T
    if accum_runs[-1] < n_runs:
        ret = np.concatenate((ret, np.array([[n_runs, accum_min_flops[-1]]])))
    return ret


def find_model_by_acc_thres(models, baseline, acc_key: str, higher_better: bool):
    valid_models = []
    for models_ in models.values():
        for model in models_:
            if higher_better ^ (model[acc_key] < baseline[acc_key]):
                valid_models.append(model)
    return max(valid_models, key=lambda d: d["fps_dnn"])


def tuning_progress(
    models,
    tuning_results_dir: str,
    baseline_fps: float,
    baseline_title: str,
    f_tuning_conf_fps,
    higher_better: bool,
):
    def add_fps_(tunings):
        for d in tunings:
            d["perf"] = f_tuning_conf_fps(models, d)
        return tunings

    tuning_results_dir_ = Path(tuning_results_dir)
    with open(tuning_results_dir_ / "unguided.pkl", "rb") as f:
        unguided = add_fps_(pkl.load(f))
    with open(tuning_results_dir_ / "guided.pkl", "rb") as f:
        guided = add_fps_(pkl.load(f))
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(*accum_best(unguided, higher_better, 50).T, label="unguided")
    ax.plot(*accum_best(guided, higher_better, 30).T, label="guided")
    ax.set_xlim(-2)
    ax.set_ylim(0)
    xl, xr = ax.get_xlim()
    ax.text(xr / 2, baseline_fps * 1.2, baseline_title, ha="center", va="center")
    ax.hlines([baseline_fps], xl, xr, colors="red", linestyles="dashed")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Model speedups


def navnet_stats(models, h_baseline, d_baseline):
    def add_to_2(axh, axd, ds, label, color):
        h_vals, d_vals = [], []
        for d in ds:
            h_vals.append((d["heading_l2"], d["fps_dnn"]))
            d_vals.append((d["distance_l2"], d["fps_dnn"]))
        axh.scatter(*zip(*h_vals), marker="x", label=label, s=8 ** 2, color=color)
        axd.scatter(*zip(*d_vals), marker="x", s=8 ** 2, color=color)

    (fig, (ax0, ax1)) = plt.subplots(1, 2, figsize=(10, 5))
    add_to_2(ax0, ax1, models["resnet18"], "ResNet18", "blue")
    add_to_2(ax0, ax1, models["squeezenet"][:12], "SqueezeNet", "red")
    add_to_2(ax0, ax1, models["darknet"], "DarkNet", "orange")
    ax0.scatter(
        h_baseline["heading_l2"],
        h_baseline["fps_dnn"],
        marker="X",
        s=14 ** 2,
        color="orange",
    )
    ax1.scatter(
        d_baseline["distance_l2"],
        d_baseline["fps_dnn"],
        marker="X",
        s=14 ** 2,
        color="orange",
    )
    ax0.set_xlabel("Heading L2 error (degree)")
    ax0.set_ylabel("FPS")
    ax0.set_xlim(0)
    ax1.set_xlabel("Distance L2 error")
    ax1.set_ylabel("FPS")
    ax1.set_xlim(0)
    # Place legend in one row at the top
    fig.tight_layout()
    fig.legend(ncol=3, bbox_to_anchor=(0.5, 0.9), loc="lower center")
    fig.savefig("navnet_models.pdf", dpi=300)


def lanenet_stats(models, baseline):
    def plot(ax, ds, label, color):
        data = [(1 - d["lane_recall"], d["fps_dnn"]) for d in ds]
        ax.scatter(*zip(*data), marker="x", label=label, s=8 ** 2, color=color)

    (fig, ax0) = plt.subplots()
    plot(ax0, models["vgg"], "VGG16", "red")
    plot(ax0, models["darknet"], "DarkNet", "blue")
    ax0.set_xlabel("Lane Det. Error")
    ax0.set_ylabel("FPS")
    ax0.set_xlim(-0.05)
    ax0.legend()
    fig.tight_layout()
    fig.savefig("lanenet_models.pdf", dpi=300)


# -----------------------------------------------------------------------------
# CropFollow results


def navnet_std_bias(models):
    def add_1(axh, ds, label, color):
        h_vals = [(d["heading_stddev"], d["heading_mean"]) for d in ds]
        axh.scatter(*zip(*h_vals), marker="x", label=label, s=8 ** 2, color=color)

    (fig, ax) = plt.subplots()
    add_1(ax, models["resnet18"], "ResNet18", "blue")
    add_1(ax, models["squeezenet"], "SqueezeNet", "red")
    add_1(ax, models["darknet"], "DarkNet", "orange")
    ax.set_xlabel("Distance Stddev")
    ax.set_ylabel("Distance Bias")
    ax.set_xlim(0)
    ax.set_ylim(0, ax.get_xlim()[1] * 0.5)
    ax.set_aspect("equal")
    # Place legend in one row at the top
    fig.legend()
    fig.tight_layout()
    fig.savefig("navnet_std_bias.pdf", dpi=300)


def cf_errorinj():
    p = Path("cropfollow/results/err_inj_mpc")
    b = np.array([0, 0])
    errorinj(
        p / "distance_fps10/searcher.pkl",
        "Bias",
        "StdDev",
        "mpc_d_10.pdf",
        lower_bound=b,
    )
    errorinj(
        p / "heading_fps6/searcher.pkl",
        "Bias (degree)",
        "StdDev (degree)",
        "mpc_h_6.pdf",
        lower_bound=b,
    )
    errorinj(
        p / "heading_fps10/searcher.pkl",
        "Bias (degree)",
        "StdDev (degree)",
        "mpc_h_10.pdf",
        lower_bound=b,
    )
    errorinj(
        p / "modelpair_fps10/searcher.pkl",
        "Heading StdDev (degree)",
        "Distance StdDev",
        "mpc_pair_10.pdf",
        figsize=(8, 4),
        lower_bound=b,
    )


def cf_tuning_progress(models, heading_baseline, distance_baseline):
    def get_tuning_conf_fps(models, d):
        dh = d["heading"]
        fpsh = models[dh["arch"]][dh["level"]]["fps_dnn"]
        dd = d["distance"]
        fpsd = models[dd["arch"]][dd["level"]]["fps_dnn"]
        return min(fpsh, fpsd)

    baseline_fps = min(heading_baseline["fps_dnn"], distance_baseline["fps_dnn"])
    fig, ax = tuning_progress(
        models,
        "cropfollow/results/autotuning",
        baseline_fps,
        "App-agnostic FPS",
        get_tuning_conf_fps,
        True,
    )
    ax.set_xlabel("Number of Field Evaluations")
    ax.set_ylabel("Best FPS")
    fig.savefig("cropfollow/results/autotuning/progress.png", dpi=300)
    fig.savefig("cf_tuning.pdf")


def gem_tuning_progress(models, baseline):
    def get_tuning_conf_util(models, d):
        req_fps = d["fps"]
        m = d["model_cfg"]
        perf_fps = models[m["arch"]][m["level"]]["fps_dnn"]
        return req_fps / perf_fps * 100

    baseline_req_fps = 10
    baseline_util = baseline_req_fps / baseline["fps_dnn"] * 100
    fig, ax = tuning_progress(
        models,
        "gem/results/autotuning",
        baseline_util,
        "App-agnostic Util.",
        get_tuning_conf_util,
        False,
    )
    ax.set_xlabel("Number of Simulator Evaluations")
    ax.set_ylabel("Lowest Util. Rate (%)")
    ax.set_ylim(0, 10)
    fig.tight_layout()
    fig.savefig("gem/results/autotuning/progress.png", dpi=300)
    fig.savefig("gem_tuning.pdf")


with open("cropfollow/results/model_list.json") as f:
    cf_models = json.load(f)
    error_base = cf_models["darknet"][0]
cf_h_baseline = find_model_by_acc_thres(cf_models, error_base, "heading_l2", False)
cf_d_baseline = find_model_by_acc_thres(cf_models, error_base, "distance_l2", False)

with open("gem/results/model_list.json") as f:
    gem_models = json.load(f)
    acc_base = gem_models["darknet"][0]
gem_baseline = find_model_by_acc_thres(gem_models, acc_base, "lane_recall", True)

# -----------------------------------------------------------------------------
# Before Eval

# navnet_std_bias(cf_models)
# navnet_stats(cf_models, cf_h_baseline, cf_d_baseline)

# # -----------------------------------------------------------------------------
# # Evaluation

# cf_tuning_progress(cf_models, cf_h_baseline, cf_d_baseline)
gem_tuning_progress(gem_models, gem_baseline)

# cf_errorinj()

# errorinj(
#     "gem/results/error_inj/searcher.pkl",
#     "FPS",
#     "Lane Det. Accuracy",
#     "gem_errorinj.pdf",
#     inverse_xy=True,
#     figsize=(8, 4),
# )

# unguided(
#     CropFollowPruneTuner,
#     "cropfollow/results/autotuning/unguided.pkl",
#     "cropfollow/results/err_inj_mpc/modelpair_fps10/searcher.pkl",
#     "cf_unguided.pdf",
#     lower_bound=np.array([0, 0]),
# )

# # -----------------------------------------------------------------------------
# # Appendix

# lanenet_stats(gem_models, gem_baseline)

# unguided(
#     GEMPruneTuner,
#     "gem/results/autotuning/unguided.pkl",
#     "gem/results/error_inj/searcher.pkl",
#     "gem_unguided.pdf",
#     inverse_xy=True,
# )
