import json
import logging
import pickle as pkl
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

PathLike = Union[Path, str]
logger = logging.getLogger(__name__)


class _SearchRegion:
    def __init__(self, p1: ndarray, p2: ndarray):
        self.p1, self.p2 = p1, p2
        assert np.all(self.p2 - self.p1 >= 0)  # p1 must be smaller than p2
        self.diag = _SearchDiag(self, 0, 1)

    @property
    def arrow(self):
        return self.p2 - self.p1

    @property
    def area(self):
        return (self.p2 - self.p1).prod()

    @property
    def region_pair(self):
        pl, pr = self.diag.end_points
        return np.array([[self.p1, pl], [pr, self.p2]])

    def get_sub_diag(self, succeeded: bool):
        t = (self.diag.t1 + self.diag.t2) / 2
        t1, t2 = (t, self.diag.t2) if succeeded else (self.diag.t1, t)
        self.diag = diag = _SearchDiag(self, t1, t2)
        return diag

    def get_sub_regions(self, threshold: np.ndarray):
        # pl is the point on the working side, pr is the point on the failing side
        (plx, ply), (prx, pry) = self.diag.end_points
        (fx, fy), (tx, ty) = self.p1, self.p2
        # Since (plx, prx) works and (ply, pry) fails, assuming convexity
        # (bx, ply) and (plx, by) should also work
        # and (tx, pry) and (prx, ty) should also fail
        inner_0, inner_1 = np.array([[fx, ply], [plx, fy]])
        outer_0, outer_1 = np.array([[tx, pry], [prx, ty]])
        # 2 pair of points to test on the next step
        # NOTE: pair them reversely. on purpose.
        region0 = _SearchRegion(inner_0, outer_1)
        region1 = _SearchRegion(inner_1, outer_0)
        return [t for t in [region0, region1] if not t.diag.within_threshold(threshold)]

    def __lt__(self, other: "_SearchRegion"):
        # 1. Not comparable (False) if both are in the same region
        if np.all(self.p1 == other.p1) and np.all(self.p1 == other.p2):
            return False
        # 2. If different regions, the one with less region area is "larger"
        # (larger means lower priority)
        if self.area != other.area:
            return self.area > other.area
        # 3. If different regions but equal area,
        # compare all the coordinates in tuples in "lexicographical" order
        our_points = (*self.p1, *self.p2)
        their_points = (*other.p1, *other.p2)
        return our_points < their_points

    def __str__(self):
        p1, p2 = self.diag.end_points
        return f"SearchRegion({self.p1} -> {self.p2}, diag=({p1} -> {p2}))"


class _SearchDiag:
    def __init__(self, parent_region: _SearchRegion, t1: float, t2: float):
        self.r = parent_region
        self.t1, self.t2 = t1, t2

    @property
    def end_points(self):
        arrow = self.r.arrow
        pl, pr = self.r.p1 + self.t1 * arrow, self.r.p1 + self.t2 * arrow
        return np.array([pl, pr])

    @property
    def mid_point(self):
        pl, pr = self.end_points
        return (pl + pr) / 2

    def within_threshold(self, thresholds: np.ndarray):
        pl, pr = self.end_points
        return np.all(pr - pl <= thresholds)


EvalT = Callable[[ndarray], float]


class Searcher:
    def __init__(
        self,
        evaluator: EvalT,
        rel_precision: float,
        n_evals: int,
        save_to: Optional[str],
    ) -> None:
        self.evaluator = evaluator
        self.rel_precision = rel_precision
        self.max_n_evals = n_evals
        self.save_to = save_to
        self._lower_bound, self._upper_bound = None, None
        self._eval_history: List[Tuple[ndarray, float]] = []
        self._boundaries: List[ndarray] = []
        self._regions: List[ndarray] = []
        self._regions_queue: List[_SearchRegion] = []

    @property
    def boundaries(self) -> ndarray:
        # 3D numpy array, (n_pairs, 2(pair), 2(dimension))
        if not self._boundaries:
            return np.zeros((0, 2, 2))
        return np.array(self._boundaries)

    @classmethod
    def load(cls, filename: str, evaluator: EvalT = None, n_eval: int = None):
        with open(filename, "rb") as f:
            instance = pkl.load(f)
        if not isinstance(instance, cls):
            raise ValueError("Loaded data is not an instance of Searcher")
        instance.evaluator = evaluator
        if n_eval is not None:
            if n_eval < instance.max_n_evals:
                raise ValueError(
                    "Cannot decrease number of evaluations in loaded cheakpoint"
                )
            instance.max_n_evals = n_eval
        return instance

    def load_json_replay(self, filename: PathLike):
        with open(filename, "r") as f:
            eval_history = json.load(f)
        # Replace evaluator and max_n_evals
        evaluator = self.evaluator
        n_evals = self.max_n_evals
        self.evaluator = _ProxyEvaluator(eval_history)
        self.max_n_evals = len(self.evaluator)
        # Replay with dummy evaluator which returns the history
        self.search_resume()
        # Restore evaluator and max_n_evals
        self.evaluator = evaluator
        self.max_n_evals = n_evals

    def save_json(self, filename: PathLike, **kwargs):
        with open(filename, "w") as f:
            eval_hist = [(point.tolist(), score) for point, score in self._eval_history]
            json.dump(eval_hist, f, **kwargs)

    def search(self, lower_bound: ndarray, upper_bound: ndarray):
        self.clear()
        self._lower_bound, self._upper_bound = lower_bound, upper_bound
        first_region = _SearchRegion(lower_bound, upper_bound)
        self._regions_queue = [first_region]
        self.search_resume()

    def search_resume(self):
        import heapq

        assert self._upper_bound is not None
        eps = self.rel_precision * (self._upper_bound - self._lower_bound)
        logger.info(f"Bisection eps of the run: {eps}")
        while self._regions_queue:
            # Peek the smallest region, don't pop.
            # This way when we save and load we can still resume on the region.
            region = self._regions_queue[0]
            diagonal = region.diag
            while not diagonal.within_threshold(eps):
                if len(self._eval_history) >= self.max_n_evals:
                    return self.boundaries
                point = diagonal.mid_point
                result = float(self.evaluator(point))
                self._eval_history.append((point, result))
                # If result > 0, this eval succeeds and we move
                # to the more aggressive part of the diagonal; vice versa.
                # (This function also updates diag of the region.)
                diagonal = region.get_sub_diag(result > 0)
                self._dump_pickle()
            # If there's no more to bisect under the given precision
            # switch to the next region.
            logger.info("Done with current region")
            # Bookkeeping
            self._boundaries.append(diagonal.end_points)
            self._regions.append(region.region_pair)
            for subregion in region.get_sub_regions(eps):
                heapq.heappush(self._regions_queue, subregion)
            # Only pop region at this point.
            heapq.heappop(self._regions_queue)
            self._dump_pickle()
        return self.boundaries

    def get_areas_stats(self) -> Dict[str, float]:
        get_sum_area = lambda point_pairs: sum(
            (p2 - p1).prod() for (p1, p2) in point_pairs
        )
        searched_area = get_sum_area(np.array(self._regions).reshape(-1, 2, 2))
        air_gap = get_sum_area(self.boundaries)
        total_area = get_sum_area(np.array([[self._lower_bound, self._upper_bound]]))
        return {
            "searched_area": searched_area,
            "precision_gap": air_gap,
            "area_to_search": total_area - searched_area - air_gap,
        }

    def plot_latest(
        self,
        ax,
        fill_rect: bool = True,
        acc_color: str = "green",
        rej_color: str = "red",
        lower_bound: Optional[ndarray] = None,
        upper_bound: Optional[ndarray] = None,
        inverse_xy: bool = False,
    ):
        if len(self._boundaries) == 0:
            return
        bound = self.boundaries.transpose((1, 0, 2))
        # Plot the decision regions
        acc_pts, rej_pts = -bound if inverse_xy else bound
        lb = lower_bound if lower_bound is not None else self._lower_bound
        ub = upper_bound if upper_bound is not None else self._upper_bound
        # When inversing, also invert the bounds and is_acc
        if inverse_xy:
            self._plot_region(ax, acc_pts, -lb, False, acc_color, fill_rect)
            self._plot_region(ax, rej_pts, -ub, True, rej_color, fill_rect)
        else:
            self._plot_region(ax, acc_pts, lb, True, acc_color, fill_rect)
            self._plot_region(ax, rej_pts, ub, False, rej_color, fill_rect)
        # Plot the boundary points
        all_points = np.array([p for p, _ in self._eval_history])
        if inverse_xy:
            all_points = -all_points
        xs, ys = all_points.T
        ax.scatter(xs, ys, marker="o", color="black", zorder=2)

    def clear(self):
        self._lower_bound, self._upper_bound = None, None
        self._eval_history = []
        self._boundaries = []
        self._regions = []
        self._regions_queue = []

    def _dump_pickle(self):
        if self.save_to is None:
            return
        with open(self.save_to, "wb") as f:
            pkl.dump(self, f)

    def _plot_region(self, ax, points, ref_pt, is_acc, color: str, fill_rect: bool):
        # Sort by x coordinates; if is accepted region (is_acc), by x increasing order
        # otherwise by x decreasing order
        x_order = np.argsort(points[:, 0])
        points = points[x_order if is_acc else x_order[::-1]]
        # Compute the prefix-max points from back to front.
        prefix_max, current_max = [], None
        for p in points[::-1]:
            cmp = lambda x, y: x > y if is_acc else x < y
            if current_max is None or cmp(p[1], current_max):
                current_max = p[1]
            prefix_max.append((p[0], current_max))
        # Reverse it so it's sorted properly by x.
        prefix_max = prefix_max[::-1]
        # Create the boundary keypoints from prefix_max points, sweeping along x axis
        boundary = []
        for (x0, y0), (_, y1) in zip(prefix_max[:-1], prefix_max[1:]):
            boundary.extend([[x0, y0], [x0, y1]])
            # x1, y1 will be inserted in the next iteration (except the last point)
        # And add the last point, and the few lines between the boundary and the ref_pt
        boundary.extend(
            [
                points[-1],
                [points[-1][0], ref_pt[1]],
                ref_pt,
                [ref_pt[0], points[0][1]],
                points[0],
            ]
        )
        xs, ys = np.array(boundary).T
        # Plot the boundary
        if fill_rect:
            ax.fill(xs, ys, color=color)
            ax.plot(xs, ys, "--", color="black")
        else:
            ax.plot(xs, ys, "-", color=color)

    def __len__(self):
        return len(self._eval_history)

    def __getstate__(self):
        return {
            "rel_precision": self.rel_precision,
            "max_n_evals": self.max_n_evals,
            "save_to": self.save_to,
            "_lower_bound": self._lower_bound,
            "_upper_bound": self._upper_bound,
            "_eval_history": self._eval_history,
            "_boundaries": self._boundaries,
            "_regions": self._regions,
            "_regions_queue": self._regions_queue,
        }


class _ProxyEvaluator:
    def __init__(self, eval_history: List[Tuple[list, float]]) -> None:
        self.points = [np.array(point) for point, _ in eval_history]
        self.scores = [score for _, score in eval_history]
        self._n_iter = 0

    def __call__(self, point: ndarray):
        assert point == self.points[self._n_iter]
        score = self.scores[self._n_iter]
        self._n_iter += 1
        return score

    def __len__(self):
        return len(self.points)
