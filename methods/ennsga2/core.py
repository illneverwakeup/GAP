from __future__ import annotations

from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import importlib
import logging
import math
import os
import platform
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools

from .config import ExperimentConfig, save_config_snapshot
from .data_io import ProblemData, read_assignment_excel, validate_problem
from .selection import EnNSGA2Selector

LOGGER = logging.getLogger("gap_ennsga2")

if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)


class EnNSGA2Experiment:
    """Модифицированный enNSGA-II для многокритериальной GAP-постановки.

    Концепция исходного скрипта сохранена:
    - хромосома: задача -> сотрудник;
    - критерии максимизации: эффективность, -ресурсы, -максимальная нагрузка;
    - repair + penalty;
    - B&B-идеальные точки и кеш;
    - модифицированная селекция по расстоянию до нормированной точки (1, 1, 1);
    - Excel-листы runs_raw, ideal_points, summary_by_params, mann_whitney_ready.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.problem: Optional[ProblemData] = None
        self.penalty_scale = 1.0
        self.ideal_efficiency: Optional[float] = None
        self.ideal_resources: Optional[float] = None
        self.ideal_workload: Optional[float] = None
        self.ideal_point_ga: Optional[Tuple[float, float, float]] = None
        self.bb_ideal_time: float = 0.0
        self.bb_nodes_total: int = 0
        self.toolbox = base.Toolbox()

    # ---------- setup ----------

    def load_problem(self) -> ProblemData:
        self.problem = read_assignment_excel(
            self.config.input.excel_path,
            self.config.input.sheet_name,
            require_each_employee_used=self.config.method.require_each_employee_used,
        )
        self.penalty_scale = self.prepare_penalty_scale()
        return self.problem

    def setup_random(self) -> None:
        if self.config.random.seed is not None:
            random.seed(self.config.random.seed)
        if self.config.random.numpy_seed is not None:
            np.random.seed(self.config.random.numpy_seed)

    def setup_toolbox(self) -> None:
        self.toolbox = base.Toolbox()
        self.toolbox.register("individualCreator", self.create_individual)
        self.toolbox.register("populationCreator", tools.initRepeat, list, self.toolbox.individualCreator)
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxOnePoint)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=self.config.ga.mutation_indpb)
        selector = EnNSGA2Selector(
            objective_func=self.calculate_raw_objectives,
            modes=("max", "max", "max"),
        )
        self.toolbox.register("select", selector.select_by_ideal_point_distance)
        self.selector = selector

    def require_problem(self) -> ProblemData:
        if self.problem is None:
            raise RuntimeError("Данные задачи ещё не загружены. Вызовите load_problem().")
        return self.problem

    # ---------- objective and constraints ----------

    def create_individual(self) -> creator.Individual:
        problem = self.require_problem()
        individual = creator.Individual(random.choices(range(problem.num_agents), k=problem.num_tasks))
        if self.config.method.use_repair:
            self.repair_individual(individual)
        return individual

    def calculate_used_resources(self, individual: List[int]) -> np.ndarray:
        problem = self.require_problem()
        assignment = np.asarray(individual, dtype=int)
        task_indices = np.arange(len(assignment))
        used_resources = np.zeros(problem.num_agents, dtype=float)
        np.add.at(used_resources, assignment, problem.r_matrix[assignment, task_indices])
        return used_resources

    def calculate_workload(self, individual: List[int]) -> np.ndarray:
        problem = self.require_problem()
        return np.bincount(np.asarray(individual, dtype=int), minlength=problem.num_agents)

    def calculate_raw_objectives(self, individual: List[int]) -> Tuple[float, float, int]:
        problem = self.require_problem()
        assignment = np.asarray(individual, dtype=int)
        task_indices = np.arange(len(assignment))
        total_efficiency = float(np.sum(problem.c_matrix[assignment, task_indices]))
        total_resources = float(np.sum(problem.r_matrix[assignment, task_indices]))
        max_workload = int(np.max(self.calculate_workload(individual))) if len(individual) else 0
        return total_efficiency, -total_resources, -max_workload

    def constraint_report(self, individual: List[int]) -> Dict[str, Any]:
        problem = self.require_problem()
        used_resources = self.calculate_used_resources(individual)
        resource_excess_by_agent = np.maximum(used_resources - problem.b_list, 0.0)
        resource_violation = float(np.sum(resource_excess_by_agent))
        workload = self.calculate_workload(individual)
        unused_employees = int(np.sum(workload == 0)) if self.config.method.require_each_employee_used else 0
        return {
            "resource_violation": resource_violation,
            "unused_employees": unused_employees,
            "is_feasible": resource_violation <= 1e-9 and unused_employees == 0,
        }

    def is_feasible(self, individual: List[int]) -> bool:
        return bool(self.constraint_report(individual)["is_feasible"])

    def calculate_penalty(self, individual: List[int]) -> float:
        if not self.config.method.use_penalty:
            return 0.0
        report = self.constraint_report(individual)
        resource_penalty = self.config.method.resource_penalty_multiplier * report["resource_violation"]
        unused_penalty = self.config.method.unused_employee_penalty_multiplier * report["unused_employees"]
        return self.penalty_scale * (resource_penalty + unused_penalty)

    def evaluate_individual(self, individual: List[int]) -> Tuple[float, float, float]:
        f1, f2, f3 = self.calculate_raw_objectives(individual)
        penalty = self.calculate_penalty(individual)
        return f1 - penalty, f2 - penalty, f3 - penalty

    # ---------- repair ----------

    def repair_individual(self, individual: List[int]) -> creator.Individual:
        if not isinstance(individual, creator.Individual):
            individual = creator.Individual(individual)
        self._repair_resource_constraints(individual)
        if self.config.method.require_each_employee_used:
            self._repair_unused_employees(individual)
            self._repair_resource_constraints(individual)
        return individual

    def _repair_resource_constraints(self, individual: List[int]) -> None:
        problem = self.require_problem()
        max_attempts = max(1, problem.num_tasks * problem.num_agents)
        for _ in range(max_attempts):
            used_resources = self.calculate_used_resources(individual)
            excess = used_resources - problem.b_list
            if np.all(excess <= 1e-9):
                return
            overloaded_agent = int(np.argmax(excess))
            overloaded_tasks = [task_idx for task_idx, agent_idx in enumerate(individual) if agent_idx == overloaded_agent]
            if not overloaded_tasks:
                return
            overloaded_tasks.sort(key=lambda task_idx: problem.r_matrix[overloaded_agent, task_idx], reverse=True)
            moved = False
            workload = self.calculate_workload(individual)
            for task_idx in overloaded_tasks:
                old_agent = individual[task_idx]
                if self.config.method.require_each_employee_used and workload[old_agent] <= 1:
                    continue
                candidates = []
                for new_agent in range(problem.num_agents):
                    if new_agent == old_agent:
                        continue
                    new_resource = used_resources[new_agent] + problem.r_matrix[new_agent, task_idx]
                    if new_resource <= problem.b_list[new_agent] + 1e-9:
                        candidates.append(new_agent)
                if not candidates:
                    continue
                new_agent = min(
                    candidates,
                    key=lambda agent_idx: (
                        used_resources[agent_idx] + problem.r_matrix[agent_idx, task_idx],
                        workload[agent_idx],
                        -problem.c_matrix[agent_idx, task_idx],
                    ),
                )
                individual[task_idx] = new_agent
                moved = True
                break
            if not moved:
                return

    def _repair_unused_employees(self, individual: List[int]) -> None:
        problem = self.require_problem()
        for _ in range(problem.num_agents):
            workload = self.calculate_workload(individual)
            unused_agents = [agent_idx for agent_idx, load in enumerate(workload) if load == 0]
            if not unused_agents:
                return
            used_resources = self.calculate_used_resources(individual)
            changed = False
            for unused_agent in unused_agents:
                donor_tasks = []
                for task_idx, donor_agent in enumerate(individual):
                    if workload[donor_agent] <= 1:
                        continue
                    if used_resources[unused_agent] + problem.r_matrix[unused_agent, task_idx] <= problem.b_list[unused_agent] + 1e-9:
                        donor_tasks.append((task_idx, donor_agent))
                if not donor_tasks:
                    continue
                task_to_move, _donor_agent = min(
                    donor_tasks,
                    key=lambda item: (
                        problem.c_matrix[item[1], item[0]] - problem.c_matrix[unused_agent, item[0]],
                        -problem.r_matrix[item[1], item[0]],
                    ),
                )
                individual[task_to_move] = unused_agent
                changed = True
            if not changed:
                return

    def repair_population(self, population: List[creator.Individual]) -> List[creator.Individual]:
        if not self.config.method.use_repair:
            return population
        repaired = []
        for individual in population:
            repaired_individual = self.repair_individual(individual)
            try:
                del repaired_individual.fitness.values
            except AttributeError:
                pass
            repaired.append(repaired_individual)
        return repaired

    def prepare_penalty_scale(self) -> float:
        problem = self.require_problem()
        max_eff_component = max(float(np.nanmax(np.abs(problem.c_matrix))), 1.0)
        max_res_component = max(float(np.nanmax(np.abs(problem.r_matrix))), 1.0)
        max_task_component = max(float(problem.num_tasks), 1.0)
        return max(max_eff_component, max_res_component, max_task_component)

    # ---------- ideal points ----------

    def dataset_key(self) -> str:
        problem = self.require_problem()
        return (
            f"{os.path.basename(str(self.config.input.excel_path))}|"
            f"sheet={self.config.input.sheet_name}|n={problem.num_agents}|m={problem.num_tasks}|"
            f"require_used={self.config.method.require_each_employee_used}"
        )

    def load_ideal_points_from_cache(self) -> bool:
        cache_path = self.config.ideal_points.cache_path
        if not self.config.ideal_points.use_cache or not os.path.exists(cache_path):
            return False
        try:
            df_cache = pd.read_excel(cache_path)
        except Exception:
            return False
        if "dataset_key" not in df_cache.columns:
            return False
        matched = df_cache[df_cache["dataset_key"] == self.dataset_key()]
        if matched.empty:
            return False
        row = matched.iloc[-1]
        self.ideal_efficiency = float(row["ideal_f1_efficiency"])
        self.ideal_resources = float(row["ideal_resources_min"])
        self.ideal_workload = float(row["ideal_workload_min"])
        self.bb_ideal_time = float(row.get("bb_ideal_time_sec", 0.0))
        self.bb_nodes_total = int(row.get("bb_nodes_total", 0))
        self.ideal_point_ga = (self.ideal_efficiency, -self.ideal_resources, -self.ideal_workload)
        LOGGER.info("Идеальные точки загружены из кеша: %s", cache_path)
        return True

    def save_ideal_points_to_cache(self) -> None:
        if not self.config.ideal_points.use_cache:
            return
        problem = self.require_problem()
        row = {
            "dataset_key": self.dataset_key(),
            "method_context": self.config.method.name,
            "file_name": os.path.basename(str(self.config.input.excel_path)),
            "sheet_name": str(self.config.input.sheet_name),
            "num_agents": problem.num_agents,
            "num_tasks": problem.num_tasks,
            "dimension": problem.dimension,
            "require_each_employee_used": self.config.method.require_each_employee_used,
            "ideal_f1_efficiency": self.ideal_efficiency,
            "ideal_resources_min": self.ideal_resources,
            "ideal_workload_min": self.ideal_workload,
            "ideal_f2_signed": -self.ideal_resources if self.ideal_resources is not None else np.nan,
            "ideal_f3_signed": -self.ideal_workload if self.ideal_workload is not None else np.nan,
            "bb_ideal_time_sec": self.bb_ideal_time,
            "bb_nodes_total": self.bb_nodes_total,
        }
        cache_path = self.config.ideal_points.cache_path
        if os.path.exists(cache_path):
            try:
                df_cache = pd.read_excel(cache_path)
            except Exception:
                df_cache = pd.DataFrame()
        else:
            df_cache = pd.DataFrame()
        if not df_cache.empty and "dataset_key" in df_cache.columns:
            df_cache = df_cache[df_cache["dataset_key"] != self.dataset_key()]
        df_cache = pd.concat([df_cache, pd.DataFrame([row])], ignore_index=True)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df_cache.to_excel(cache_path, index=False)

    def compute_ideal_points_with_bb(self) -> None:
        if self.load_ideal_points_from_cache():
            return
        if not self.config.ideal_points.use_bb:
            raise RuntimeError(
                "Идеальные точки не найдены в кеше, а расчёт B&B отключён. "
                "Включите ideal_points.use_bb=true или укажите существующий cache_path."
            )

        try:
            bnb_module = importlib.import_module("branch_and_bound_assignment_1")
        except ImportError as exc:
            raise RuntimeError(
                "Модуль branch_and_bound_assignment_1.py не найден. "
                "Поместите его рядом с запускным скриптом или отключите расчёт B&B "
                "и используйте готовый кеш идеальных точек."
            ) from exc

        problem = self.require_problem()
        if hasattr(bnb_module, "REQUIRE_EACH_EMPLOYEE_USED"):
            bnb_module.REQUIRE_EACH_EMPLOYEE_USED = self.config.method.require_each_employee_used

        LOGGER.info("Расчёт идеальных точек методом ветвей и границ")
        start_time = time.time()
        result_f1 = bnb_module.branch_and_bound(problem.c_matrix.tolist(), problem.r_matrix.tolist(), problem.b_list.tolist(), objective_type=1)
        result_f2 = bnb_module.branch_and_bound(problem.c_matrix.tolist(), problem.r_matrix.tolist(), problem.b_list.tolist(), objective_type=2)
        result_f3 = bnb_module.branch_and_bound(problem.c_matrix.tolist(), problem.r_matrix.tolist(), problem.b_list.tolist(), objective_type=3)
        self.bb_ideal_time = time.time() - start_time
        self.bb_nodes_total = int(result_f1.nodes_visited + result_f2.nodes_visited + result_f3.nodes_visited)
        self.ideal_efficiency = float(result_f1.total_efficiency)
        self.ideal_resources = float(result_f2.total_resources)
        self.ideal_workload = float(result_f3.max_workload)
        self.ideal_point_ga = (self.ideal_efficiency, -self.ideal_resources, -self.ideal_workload)
        self.save_ideal_points_to_cache()

    # ---------- metrics ----------

    @staticmethod
    def safe_percent_deviation(value: float, ideal_value: Optional[float]) -> float:
        if ideal_value is None or abs(ideal_value) < 1e-12:
            return float("nan")
        return abs((value - ideal_value) / ideal_value) * 100.0

    def calculate_population_ranges(self, individuals: List[creator.Individual]) -> Dict[str, float]:
        values = []
        for individual in individuals:
            raw_f1, raw_f2, raw_f3 = self.calculate_raw_objectives(individual)
            values.append((float(raw_f1), float(-raw_f2), float(-raw_f3)))
        if not values:
            return {k: float("nan") for k in [
                "f1_min", "f1_max", "resources_min", "resources_max", "workload_min", "workload_max"
            ]}
        arr = np.asarray(values, dtype=float)
        return {
            "f1_min": float(np.min(arr[:, 0])),
            "f1_max": float(np.max(arr[:, 0])),
            "resources_min": float(np.min(arr[:, 1])),
            "resources_max": float(np.max(arr[:, 1])),
            "workload_min": float(np.min(arr[:, 2])),
            "workload_max": float(np.max(arr[:, 2])),
        }

    @staticmethod
    def safe_max_norm(value: float, min_value: float, max_value: float) -> float:
        if any(math.isnan(v) for v in (value, min_value, max_value)):
            return float("nan")
        if abs(max_value - min_value) < 1e-12:
            return 1.0
        return (value - min_value) / (max_value - min_value)

    @staticmethod
    def safe_min_norm(value: float, min_value: float, max_value: float) -> float:
        if any(math.isnan(v) for v in (value, min_value, max_value)):
            return float("nan")
        if abs(max_value - min_value) < 1e-12:
            return 1.0
        return (max_value - value) / (max_value - min_value)

    def calculate_distance_to_111(self, raw_f1: float, raw_f2: float, raw_f3: float, ranges: Dict[str, float]) -> Dict[str, float]:
        f1_efficiency = float(raw_f1)
        resources_total = float(-raw_f2)
        max_workload = float(-raw_f3)
        z1 = self.safe_max_norm(f1_efficiency, ranges["f1_min"], ranges["f1_max"])
        z2 = self.safe_min_norm(resources_total, ranges["resources_min"], ranges["resources_max"])
        z3 = self.safe_min_norm(max_workload, ranges["workload_min"], ranges["workload_max"])
        distance_to_111 = float("nan") if any(math.isnan(v) for v in (z1, z2, z3)) else math.sqrt((1.0 - z1) ** 2 + (1.0 - z2) ** 2 + (1.0 - z3) ** 2)
        return {
            "z1_efficiency_norm": z1,
            "z2_resources_norm": z2,
            "z3_workload_norm": z3,
            "distance_to_111": distance_to_111,
            "norm_f1_min": ranges["f1_min"],
            "norm_f1_max": ranges["f1_max"],
            "norm_resources_min": ranges["resources_min"],
            "norm_resources_max": ranges["resources_max"],
            "norm_workload_min": ranges["workload_min"],
            "norm_workload_max": ranges["workload_max"],
        }

    def calculate_analysis_metrics(self, raw_f1: float, raw_f2: float, raw_f3: float, normalization_ranges: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        f1_efficiency = float(raw_f1)
        resources_total = float(-raw_f2)
        max_workload = float(-raw_f3)
        dev_f1 = self.safe_percent_deviation(f1_efficiency, self.ideal_efficiency)
        dev_f2 = self.safe_percent_deviation(resources_total, self.ideal_resources)
        dev_f3 = self.safe_percent_deviation(max_workload, self.ideal_workload)
        deviations = np.asarray([dev_f1, dev_f2, dev_f3], dtype=float)
        dev_avg = float(np.nanmean(deviations))
        dev_std = float(np.nanstd(deviations))
        if (
            self.ideal_efficiency is None or abs(self.ideal_efficiency) < 1e-12 or
            self.ideal_resources is None or abs(self.ideal_resources) < 1e-12 or
            self.ideal_workload is None or abs(self.ideal_workload) < 1e-12
        ):
            distance_to_ideal = float("nan")
        else:
            distance_to_ideal = math.sqrt(
                ((self.ideal_efficiency - f1_efficiency) / abs(self.ideal_efficiency)) ** 2 +
                ((resources_total - self.ideal_resources) / abs(self.ideal_resources)) ** 2 +
                ((max_workload - self.ideal_workload) / abs(self.ideal_workload)) ** 2
            )
        if normalization_ranges is None:
            distance_111_metrics = {k: float("nan") for k in [
                "z1_efficiency_norm", "z2_resources_norm", "z3_workload_norm", "distance_to_111",
                "norm_f1_min", "norm_f1_max", "norm_resources_min", "norm_resources_max", "norm_workload_min", "norm_workload_max"
            ]}
        else:
            distance_111_metrics = self.calculate_distance_to_111(raw_f1, raw_f2, raw_f3, normalization_ranges)
        return {
            "f1_efficiency": f1_efficiency,
            "f2_resources_signed": float(raw_f2),
            "f3_workload_signed": float(raw_f3),
            "resources_total": resources_total,
            "max_workload": max_workload,
            "ideal_f1_efficiency": self.ideal_efficiency,
            "ideal_resources_min": self.ideal_resources,
            "ideal_workload_min": self.ideal_workload,
            "ideal_f2_signed": -self.ideal_resources if self.ideal_resources is not None else np.nan,
            "ideal_f3_signed": -self.ideal_workload if self.ideal_workload is not None else np.nan,
            "dev_f1_percent": dev_f1,
            "dev_f2_percent": dev_f2,
            "dev_f3_percent": dev_f3,
            "dev_avg_percent": dev_avg,
            "dev_std_percent": dev_std,
            "distance_to_ideal": distance_to_ideal,
            **distance_111_metrics,
        }

    # ---------- algorithm ----------

    @staticmethod
    def nsga2_parent_selection(population: List[creator.Individual], k: int) -> List[creator.Individual]:
        if k <= 0:
            return []
        if len(population) < 4 or k < 4:
            return random.choices(population, k=k)
        if k % 4 == 0:
            return list(tools.selTournamentDCD(population, k))
        main_count = k - (k % 4)
        parents = list(tools.selTournamentDCD(population, main_count)) if main_count > 0 else []
        extra = list(tools.selTournamentDCD(population, 4))
        parents.extend(extra[: k - main_count])
        return parents

    def evaluate_population(self, population: List[creator.Individual]) -> None:
        invalid_individuals = [individual for individual in population if not individual.fitness.valid]
        fitness_values = list(map(self.toolbox.evaluate, invalid_individuals))
        for individual, fitness in zip(invalid_individuals, fitness_values):
            individual.fitness.values = fitness

    def collect_result_rows(self, individuals: List[creator.Individual], repeat: int, pop_size: int, p_cross: float, p_mut: float, max_gen: int, elapsed_time: float, normalization_ranges: Optional[Dict[str, float]] = None) -> List[dict]:
        problem = self.require_problem()
        rows = []
        added_individuals = set()
        for solution_index, individual in enumerate(individuals, start=1):
            individual_tuple = tuple(individual)
            if individual_tuple in added_individuals:
                continue
            raw_f1, raw_f2, raw_f3 = self.calculate_raw_objectives(individual)
            analysis_metrics = self.calculate_analysis_metrics(raw_f1, raw_f2, raw_f3, normalization_ranges)
            penalty = self.calculate_penalty(individual)
            report = self.constraint_report(individual)
            used_resources = self.calculate_used_resources(individual)
            workload = self.calculate_workload(individual)
            if self.config.method.save_only_feasible_results and not report["is_feasible"]:
                continue
            run_id = (
                f"{self.config.method.name}|{os.path.basename(str(self.config.input.excel_path))}|"
                f"rep={repeat}|pop={pop_size}|cx={p_cross}|mut={p_mut}|gen={max_gen}"
            )
            rows.append({
                "method": self.config.method.name,
                "file_name": os.path.basename(str(self.config.input.excel_path)),
                "sheet_name": str(self.config.input.sheet_name),
                "num_agents": problem.num_agents,
                "num_tasks": problem.num_tasks,
                "dimension": problem.dimension,
                "repeat": repeat,
                "run_id": run_id,
                "solution_index_in_front": solution_index,
                "pop_size": pop_size,
                "crossover_rate": p_cross,
                "mutation_rate": p_mut,
                "num_generations": max_gen,
                "mutation_indpb": self.config.ga.mutation_indpb,
                "require_each_employee_used": self.config.method.require_each_employee_used,
                "use_repair": self.config.method.use_repair,
                "use_penalty": self.config.method.use_penalty,
                "random_seed": self.config.random.seed,
                "numpy_seed": self.config.random.numpy_seed,
                "config_hash": self.config.hash(),
                "chromosome": str(list(individual)),
                **analysis_metrics,
                "fitness_f1_with_penalty": float(individual.fitness.values[0]),
                "fitness_f2_with_penalty": float(individual.fitness.values[1]),
                "fitness_f3_with_penalty": float(individual.fitness.values[2]),
                "penalty": float(penalty),
                "resource_violation": float(report["resource_violation"]),
                "unused_employees": int(report["unused_employees"]),
                "is_feasible": bool(report["is_feasible"]),
                "used_resources_by_employee": str(used_resources.tolist()),
                "workload_by_employee": str(workload.tolist()),
                "elapsed_time_sec": float(elapsed_time),
                "bb_ideal_time_sec": float(self.bb_ideal_time),
                "bb_nodes_total": int(self.bb_nodes_total),
            })
            added_individuals.add(individual_tuple)
        return rows

    def run(self) -> str:
        self.setup_random()
        problem = self.load_problem()
        validate_problem(problem, self.config.method.require_each_employee_used)
        self.setup_toolbox()
        self.compute_ideal_points_with_bb()
        LOGGER.info("Данные загружены: n=%s, m=%s, dimension=%s", problem.num_agents, problem.num_tasks, problem.dimension)
        LOGGER.info("Метод: enNSGA-II, локальная min-max нормализация и селекция по D111")

        all_results: List[dict] = []
        for repeat in range(1, self.config.ga.repeats + 1):
            LOGGER.info("Итерация %s/%s", repeat, self.config.ga.repeats)
            for pop_size, p_cross, p_mut, max_gen in product(
                self.config.ga.pop_sizes,
                self.config.ga.crossover_rates,
                self.config.ga.mutation_rates,
                self.config.ga.generations,
            ):
                LOGGER.info("Параметры: pop=%s, cx=%s, mut=%s, gen=%s", pop_size, p_cross, p_mut, max_gen)
                start_time = time.time()

                population = self.toolbox.populationCreator(n=pop_size)
                population = self.repair_population(population)
                self.evaluate_population(population)

                for generation in range(max_gen):
                    # В отличие от классического NSGA-II, здесь используется
                    # модифицированный отбор: из P ∪ Q выбираются решения с
                    # минимальным расстоянием до локально нормированной точки (1, 1, 1).
                    offspring = algorithms.varAnd(population, self.toolbox, p_cross, p_mut)
                    offspring = self.repair_population(offspring)
                    self.evaluate_population(offspring)
                    population = self.toolbox.select(population + offspring, pop_size)

                    if self.config.output.verbose and max_gen >= 50 and (generation + 1) % max(1, max_gen // 5) == 0:
                        LOGGER.info("  поколение %s/%s", generation + 1, max_gen)

                final_ranges = self.calculate_population_ranges(population)
                fronts = self.selector.euclidean_ideal_point_sort(population, pop_size)
                candidate_individuals = fronts[0] if fronts else []
                elapsed_time = time.time() - start_time

                result_rows = self.collect_result_rows(
                    candidate_individuals,
                    repeat,
                    pop_size,
                    p_cross,
                    p_mut,
                    max_gen,
                    elapsed_time,
                    final_ranges,
                )

                if not result_rows and self.config.method.save_only_feasible_results:
                    feasible_population = [individual for individual in population if self.is_feasible(individual)]
                    result_rows = self.collect_result_rows(feasible_population, repeat, pop_size, p_cross, p_mut, max_gen, elapsed_time)

                if not result_rows:
                    LOGGER.warning("Не удалось сохранить допустимые решения для этого запуска.")
                else:
                    all_results.extend(result_rows)

        df_results = pd.DataFrame(all_results)
        return self.save_results_to_excel(df_results)

    # ---------- export ----------

    def build_output_tables(self, df_results: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        problem = self.require_problem()
        now = datetime.now().isoformat(timespec="seconds")
        df_ideal = pd.DataFrame([{
            "method": self.config.method.name,
            "file_name": os.path.basename(str(self.config.input.excel_path)),
            "sheet_name": str(self.config.input.sheet_name),
            "num_agents": problem.num_agents,
            "num_tasks": problem.num_tasks,
            "dimension": problem.dimension,
            "require_each_employee_used": self.config.method.require_each_employee_used,
            "ideal_f1_efficiency": self.ideal_efficiency,
            "ideal_resources_min": self.ideal_resources,
            "ideal_workload_min": self.ideal_workload,
            "ideal_f2_signed": -self.ideal_resources if self.ideal_resources is not None else np.nan,
            "ideal_f3_signed": -self.ideal_workload if self.ideal_workload is not None else np.nan,
            "bb_ideal_time_sec": self.bb_ideal_time,
            "bb_nodes_total": self.bb_nodes_total,
        }])

        metadata = {
            "created_at": now,
            "method": self.config.method.name,
            "input_file": self.config.input.excel_path,
            "sheet_name": self.config.input.sheet_name,
            "num_agents": problem.num_agents,
            "num_tasks": problem.num_tasks,
            "dimension": problem.dimension,
            "config_hash": self.config.hash(),
            "random_seed": self.config.random.seed,
            "numpy_seed": self.config.random.numpy_seed,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        }
        try:
            import deap  # type: ignore
            metadata["deap_version"] = deap.__version__
        except Exception:
            metadata["deap_version"] = "unknown"
        df_metadata = pd.DataFrame([metadata])

        config_rows = []
        def flatten(prefix: str, obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    flatten(f"{prefix}.{k}" if prefix else k, v)
            else:
                config_rows.append({"parameter": prefix, "value": str(obj)})
        flatten("", self.config.to_dict())
        df_config = pd.DataFrame(config_rows)

        if df_results.empty:
            return df_ideal, pd.DataFrame(), pd.DataFrame(), df_metadata, df_config

        group_cols = [
            "method", "file_name", "num_agents", "num_tasks", "dimension",
            "pop_size", "crossover_rate", "mutation_rate", "num_generations",
        ]
        df_summary = (
            df_results
            .groupby(group_cols, dropna=False)
            .agg(
                solutions_count=("dev_avg_percent", "count"),
                feasible_count=("is_feasible", "sum"),
                mean_dev_avg_percent=("dev_avg_percent", "mean"),
                median_dev_avg_percent=("dev_avg_percent", "median"),
                std_dev_avg_percent=("dev_avg_percent", "std"),
                min_dev_avg_percent=("dev_avg_percent", "min"),
                max_dev_avg_percent=("dev_avg_percent", "max"),
                mean_dev_std_percent=("dev_std_percent", "mean"),
                median_dev_std_percent=("dev_std_percent", "median"),
                min_dev_std_percent=("dev_std_percent", "min"),
                mean_distance_to_ideal=("distance_to_ideal", "mean"),
                median_distance_to_ideal=("distance_to_ideal", "median"),
                std_distance_to_ideal=("distance_to_ideal", "std"),
                min_distance_to_ideal=("distance_to_ideal", "min"),
                mean_distance_to_111=("distance_to_111", "mean"),
                median_distance_to_111=("distance_to_111", "median"),
                std_distance_to_111=("distance_to_111", "std"),
                min_distance_to_111=("distance_to_111", "min"),
                mean_elapsed_time_sec=("elapsed_time_sec", "mean"),
                min_elapsed_time_sec=("elapsed_time_sec", "min"),
            )
            .reset_index()
        )

        mw_group_cols = group_cols + ["repeat"]
        idx = df_results.groupby(mw_group_cols, dropna=False)["distance_to_ideal"].idxmin()
        df_mann_whitney = df_results.loc[idx].copy().sort_values(mw_group_cols)
        df_mann_whitney["best_distance_to_ideal"] = df_mann_whitney["distance_to_ideal"]
        df_mann_whitney["best_distance_to_111"] = df_mann_whitney["distance_to_111"]
        df_mann_whitney["best_dev_avg_percent"] = df_mann_whitney["dev_avg_percent"]
        df_mann_whitney["best_dev_std_percent"] = df_mann_whitney["dev_std_percent"]
        df_mann_whitney["best_elapsed_time_sec"] = df_mann_whitney["elapsed_time_sec"]
        df_mann_whitney["mann_whitney_sample_value"] = df_mann_whitney["best_distance_to_ideal"]
        df_mann_whitney["mann_whitney_metric"] = "distance_to_ideal"
        df_mann_whitney["alternative_hypothesis"] = "enNSGAII_less_than_NSGAII"
        df_mann_whitney["sample_unit"] = "best_solution_per_repeat_by_min_distance_to_ideal"
        preferred_cols = [
            "method", "file_name", "repeat", "pop_size", "crossover_rate",
            "mutation_rate", "num_generations", "best_distance_to_ideal",
            "best_distance_to_111", "best_dev_avg_percent", "best_dev_std_percent", "best_elapsed_time_sec",
            "mann_whitney_sample_value", "mann_whitney_metric", "alternative_hypothesis", "sample_unit",
        ]
        remaining_cols = [col for col in df_mann_whitney.columns if col not in preferred_cols]
        df_mann_whitney = df_mann_whitney[preferred_cols + remaining_cols]
        return df_ideal, df_summary, df_mann_whitney, df_metadata, df_config

    def save_results_to_excel(self, df_results: pd.DataFrame) -> str:
        problem = self.require_problem()
        output_dir = Path(self.config.output.directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.config.output.prefix}_{problem.num_agents}x{problem.num_tasks}.xlsx"
        df_ideal, df_summary, df_mann_whitney, df_metadata, df_config = self.build_output_tables(df_results)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_results.to_excel(writer, sheet_name="runs_raw", index=False)
            df_ideal.to_excel(writer, sheet_name="ideal_points", index=False)
            df_summary.to_excel(writer, sheet_name="summary_by_params", index=False)
            df_mann_whitney.to_excel(writer, sheet_name="mann_whitney_ready", index=False)
            df_metadata.to_excel(writer, sheet_name="metadata", index=False)
            df_config.to_excel(writer, sheet_name="config", index=False)
        snapshot = output_dir / f"{self.config.output.prefix}_{problem.num_agents}x{problem.num_tasks}_{self.config.hash()}_config.yaml"
        save_config_snapshot(self.config, str(snapshot))
        LOGGER.info("Результаты сохранены: %s", output_path)
        return str(output_path)


def run_experiment(config: ExperimentConfig) -> str:
    experiment = EnNSGA2Experiment(config)
    return experiment.run()
