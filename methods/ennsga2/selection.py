"""
Reusable enNSGA-II selection operators.

The main function/class in this module implements local min-max normalization
and ranks individuals by Euclidean distance to the normalized ideal point
(1, 1, 1, ...). It is designed to be registered in a DEAP toolbox in the
same style as tools.selNSGA2:

    from ennsga2_selection import EnNSGA2Selector

    selector = EnNSGA2Selector(
        objective_func=calculate_raw_objectives,
        modes=("max", "max", "max"),  # if calculate_raw_objectives returns f1, -resources, -workload
    )
    toolbox.register("select", selector.select_by_ideal_point_distance)

If objective_func is not passed, the selector uses individual.fitness.values.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ObjectiveFunc = Callable[[object], Sequence[float]]


class EnNSGA2Selector:
    """
    Reusable enNSGA-II selector based on distance to the normalized ideal point.

    Parameters
    ----------
    objective_func:
        Function that receives an individual and returns objective values.
        If None, the selector uses individual.fitness.values.

    modes:
        Optimization direction for each objective: "max" or "min".
        After local min-max normalization, every objective is transformed so
        that larger normalized values are better and the ideal point is
        (1, 1, ..., 1).

        Examples:
        - objective_func returns (efficiency, -resources, -workload):
          modes=("max", "max", "max")
        - objective_func returns (efficiency, resources, workload):
          modes=("max", "min", "min")

    eps:
        Tolerance used to detect constant objective ranges.
    """

    def __init__(
        self,
        objective_func: Optional[ObjectiveFunc] = None,
        modes: Sequence[str] = ("max", "max", "max"),
        eps: float = 1e-12,
    ) -> None:
        self.objective_func = objective_func
        self.modes = tuple(mode.lower() for mode in modes)
        self.eps = eps
        self._validate_modes()

    def _validate_modes(self) -> None:
        invalid = [mode for mode in self.modes if mode not in {"max", "min"}]
        if invalid:
            raise ValueError(f"Unsupported objective modes: {invalid}. Use only 'max' or 'min'.")

    def _objective_values(self, individuals: Sequence[object]) -> np.ndarray:
        if not individuals:
            return np.empty((0, len(self.modes)), dtype=float)

        if self.objective_func is None:
            values = [tuple(individual.fitness.values) for individual in individuals]
        else:
            values = [tuple(self.objective_func(individual)) for individual in individuals]

        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Objective values must form a 2D numeric array.")
        if arr.shape[1] != len(self.modes):
            raise ValueError(
                f"Objective count ({arr.shape[1]}) does not match modes count ({len(self.modes)})."
            )
        return arr

    def normalize(self, objective_values: np.ndarray) -> np.ndarray:
        """
        Locally min-max normalizes objectives so that 1 is best for every criterion.
        """
        if objective_values.size == 0:
            return objective_values.copy()

        mins = np.nanmin(objective_values, axis=0)
        maxs = np.nanmax(objective_values, axis=0)
        ranges = maxs - mins

        normalized = np.ones_like(objective_values, dtype=float)
        non_constant = np.abs(ranges) > self.eps

        for col, mode in enumerate(self.modes):
            if not non_constant[col]:
                normalized[:, col] = 1.0
                continue
            if mode == "max":
                normalized[:, col] = (objective_values[:, col] - mins[col]) / ranges[col]
            else:  # mode == "min"
                normalized[:, col] = (maxs[col] - objective_values[:, col]) / ranges[col]

        return normalized

    @staticmethod
    def distance_to_ideal(normalized_values: np.ndarray) -> np.ndarray:
        """Returns Euclidean distances to the normalized ideal point (1, ..., 1)."""
        if normalized_values.size == 0:
            return np.empty(0, dtype=float)
        return np.sqrt(np.sum((1.0 - normalized_values) ** 2, axis=1))

    def distances(self, individuals: Sequence[object]) -> np.ndarray:
        """Computes local normalized distances for a sequence of individuals."""
        values = self._objective_values(individuals)
        normalized = self.normalize(values)
        return self.distance_to_ideal(normalized)

    def select_by_ideal_point_distance(self, individuals: Sequence[object], k: int) -> List[object]:
        """
        DEAP-compatible selector: returns k individuals with the smallest distance.
        """
        individuals = list(individuals)
        if k <= 0 or not individuals:
            return []

        k = min(k, len(individuals))
        distances = self.distances(individuals)

        # argpartition avoids full sorting when k < len(individuals).
        if k < len(individuals):
            best_idx = np.argpartition(distances, k - 1)[:k]
            best_idx = best_idx[np.argsort(distances[best_idx], kind="stable")]
        else:
            best_idx = np.argsort(distances, kind="stable")

        return [individuals[int(i)] for i in best_idx]

    def euclidean_ideal_point_sort(self, individuals: Sequence[object], k: int) -> List[List[object]]:
        """
        Returns pseudo-fronts ordered by increasing distance to (1, ..., 1).

        This mirrors the old project API used by the enNSGA-II scripts. Distances
        are grouped by exact equal values, preserving the conceptual behavior of
        the previous implementation.
        """
        individuals = list(individuals)
        if k <= 0 or not individuals:
            return []

        k = min(k, len(individuals))
        distances = self.distances(individuals)
        order = np.argsort(distances, kind="stable")[:k]

        fronts: List[List[object]] = []
        current_front: List[object] = []
        current_distance: Optional[float] = None

        for idx in order:
            distance = float(distances[int(idx)])
            individual = individuals[int(idx)]
            if current_distance is None or distance == current_distance:
                current_front.append(individual)
            else:
                fronts.append(current_front)
                current_front = [individual]
            current_distance = distance

        if current_front:
            fronts.append(current_front)

        return fronts


# Function-style wrapper, similar in spirit to DEAP tools.selNSGA2.
def selEnNSGA2(
    individuals: Sequence[object],
    k: int,
    objective_func: Optional[ObjectiveFunc] = None,
    modes: Sequence[str] = ("max", "max", "max"),
    eps: float = 1e-12,
) -> List[object]:
    selector = EnNSGA2Selector(objective_func=objective_func, modes=modes, eps=eps)
    return selector.select_by_ideal_point_distance(individuals, k)
