from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import logging
import math
import time

import numpy as np
import pandas as pd

try:
    from .data_io import ProblemData, read_assignment_excel as _read_problem_excel, validate_problem
except Exception:  # pragma: no cover - allows legacy single-file fallback if copied alone
    ProblemData = None  # type: ignore
    _read_problem_excel = None  # type: ignore
    validate_problem = None  # type: ignore

LOGGER = logging.getLogger(__name__)

# Legacy-compatible global switch. NSGA-II/enNSGA-II scripts can still set it
# before calling branch_and_bound(...), exactly as in the original script.
REQUIRE_EACH_EMPLOYEE_USED = True

OBJECTIVE_NAMES: Dict[int, str] = {
    1: "Максимизация эффективности назначения",
    2: "Минимизация суммарных ресурсов",
    3: "Минимизация максимальной нагрузки сотрудника",
}

OBJECTIVE_CODES: Dict[str, int] = {
    "efficiency": 1,
    "resources": 2,
    "workload": 3,
    "f1": 1,
    "f2": 2,
    "f3": 3,
    "1": 1,
    "2": 2,
    "3": 3,
}


@dataclass
class AssignmentResult:
    objective_type: int
    objective_name: str
    best_value: float
    assignment: List[int]
    total_efficiency: float
    total_resources: float
    max_workload: int
    used_resources_by_employee: List[float]
    workload_by_employee: List[int]
    elapsed_seconds: float
    nodes_visited: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def parse_objective(value: Union[int, str]) -> int:
    """Converts objective aliases to the numeric objective code used in the article."""
    if isinstance(value, int):
        objective = value
    else:
        key = str(value).strip().lower()
        if key not in OBJECTIVE_CODES:
            raise ValueError(
                "Неизвестная целевая функция. Используйте 1/2/3, "
                "efficiency/resources/workload или all в CLI."
            )
        objective = OBJECTIVE_CODES[key]
    if objective not in OBJECTIVE_NAMES:
        raise ValueError("objective_type должен быть 1, 2 или 3.")
    return objective


def _to_lists(
    C: Sequence[Sequence[float]],
    R: Sequence[Sequence[float]],
    b: Sequence[float],
) -> Tuple[List[List[float]], List[List[float]], List[float]]:
    C_list = [[float(x) for x in row] for row in C]
    R_list = [[float(x) for x in row] for row in R]
    b_list = [float(x) for x in b]
    return C_list, R_list, b_list


def validate_assignment_data(
    C: Sequence[Sequence[float]],
    R: Sequence[Sequence[float]],
    b: Sequence[float],
    require_each_employee_used: bool = True,
) -> None:
    """Validates GAP matrices for the B&B solver and GA ideal-point calls."""
    C_list, R_list, b_list = _to_lists(C, R, b)
    n = len(C_list)
    if n == 0:
        raise ValueError("Матрица эффективности C пустая.")
    m = len(C_list[0])
    if m == 0:
        raise ValueError("В матрице эффективности C нет задач.")
    if len(R_list) != n:
        raise ValueError("Количество строк в R должно совпадать с количеством сотрудников.")
    if len(b_list) != n:
        raise ValueError("Длина вектора b должна совпадать с количеством сотрудников.")
    if require_each_employee_used and m < n:
        raise ValueError(
            "Невозможно назначить каждому сотруднику хотя бы одну задачу: "
            "количество задач меньше количества сотрудников."
        )
    for i in range(n):
        if len(C_list[i]) != m:
            raise ValueError(f"Строка {i + 1} матрицы C имеет неправильную длину.")
        if len(R_list[i]) != m:
            raise ValueError(f"Строка {i + 1} матрицы R имеет неправильную длину.")
        if b_list[i] < 0:
            raise ValueError("Ресурсные ограничения b не должны быть отрицательными.")
        for j in range(m):
            if math.isnan(C_list[i][j]) or math.isnan(R_list[i][j]):
                raise ValueError("Во входных данных обнаружены пустые или некорректные числовые значения.")
            if R_list[i][j] < 0:
                raise ValueError("Ресурсы R не должны быть отрицательными.")
    for j in range(m):
        if not any(R_list[i][j] <= b_list[i] + 1e-9 for i in range(n)):
            raise ValueError(f"Задачу {j + 1} невозможно назначить ни одному сотруднику.")


def read_assignment_excel(
    path: str,
    sheet_name: Union[int, str] = 0,
    require_each_employee_used: bool = True,
) -> Tuple[List[List[float]], List[List[float]], List[float]]:
    """Reads the common GAP Excel format and returns C, R, b as Python lists."""
    if _read_problem_excel is not None:
        problem = _read_problem_excel(path, sheet_name, require_each_employee_used=require_each_employee_used)
        return problem.c_matrix.tolist(), problem.r_matrix.tolist(), problem.b_list.tolist()

    df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    n = int(df.iloc[0, 0])
    m = int(df.iloc[2, 0])
    c_start_row = 4
    r_start_row = c_start_row + n + 1
    b_start_row = r_start_row + n + 1
    C = df.iloc[c_start_row:c_start_row + n, 0:m].astype(float).values.tolist()
    R = df.iloc[r_start_row:r_start_row + n, 0:m].astype(float).values.tolist()
    b = df.iloc[b_start_row:b_start_row + n, 0].astype(float).values.tolist()
    validate_assignment_data(C, R, b, require_each_employee_used=require_each_employee_used)
    return C, R, b


def evaluate_assignment(
    assignment: Sequence[int],
    C: Sequence[Sequence[float]],
    R: Sequence[Sequence[float]],
    b: Sequence[float],
) -> Tuple[float, float, int, List[float], List[int]]:
    C_list, R_list, b_list = _to_lists(C, R, b)
    n = len(C_list)
    m = len(C_list[0])
    if len(assignment) != m:
        raise ValueError(f"Длина assignment должна быть равна числу задач: ожидалось {m}.")

    total_efficiency = 0.0
    total_resources = 0.0
    used_resources = [0.0] * n
    workload = [0] * n

    for j, employee in enumerate(assignment):
        i = int(employee)
        if i < 0 or i >= n:
            raise ValueError(f"Некорректный индекс сотрудника {i} для задачи {j + 1}.")
        total_efficiency += C_list[i][j]
        total_resources += R_list[i][j]
        used_resources[i] += R_list[i][j]
        workload[i] += 1

    for i in range(n):
        if used_resources[i] > b_list[i] + 1e-9:
            raise ValueError("Полученное решение нарушает ресурсные ограничения.")

    return total_efficiency, total_resources, max(workload) if workload else 0, used_resources, workload


def objective_value(
    assignment: Sequence[int],
    C: Sequence[Sequence[float]],
    R: Sequence[Sequence[float]],
    b: Sequence[float],
    objective_type: Union[int, str],
) -> float:
    objective = parse_objective(objective_type)
    total_eff, total_res, max_load, _, _ = evaluate_assignment(assignment, C, R, b)
    if objective == 1:
        return total_eff
    if objective == 2:
        return total_res
    return float(max_load)


def branch_and_bound(
    C: Sequence[Sequence[float]],
    R: Sequence[Sequence[float]],
    b: Sequence[float],
    objective_type: Union[int, str],
    require_each_employee_used: Optional[bool] = None,
) -> AssignmentResult:
    """
    Solves the GAP assignment problem by branch and bound.

    objective_type:
      1 / efficiency — maximize total efficiency;
      2 / resources  — minimize total resources;
      3 / workload   — minimize maximum employee workload.

    The function remains backward-compatible with the original scripts: if
    require_each_employee_used is omitted, the module-level
    REQUIRE_EACH_EMPLOYEE_USED flag is used.
    """
    objective = parse_objective(objective_type)
    require_used = REQUIRE_EACH_EMPLOYEE_USED if require_each_employee_used is None else bool(require_each_employee_used)
    C_list, R_list, b_list = _to_lists(C, R, b)
    validate_assignment_data(C_list, R_list, b_list, require_each_employee_used=require_used)

    start_time = time.time()
    n = len(C_list)
    m = len(C_list[0])

    feasible_employees_for_task: List[List[int]] = []
    for j in range(m):
        feasible = [i for i in range(n) if R_list[i][j] <= b_list[i] + 1e-9]
        if not feasible:
            raise ValueError(f"Задачу {j + 1} невозможно назначить ни одному сотруднику.")
        feasible_employees_for_task.append(feasible)

    if objective == 1:
        for j in range(m):
            feasible_employees_for_task[j].sort(key=lambda i: C_list[i][j], reverse=True)
    elif objective == 2:
        for j in range(m):
            feasible_employees_for_task[j].sort(key=lambda i: R_list[i][j])

    best_value = -math.inf if objective == 1 else math.inf
    best_assignment: Optional[List[int]] = None
    nodes_visited = 0

    current_assignment = [-1] * m
    used_resources = [0.0] * n
    workload = [0] * n

    suffix_max_eff = [0.0] * (m + 1)
    if objective == 1:
        for j in range(m - 1, -1, -1):
            suffix_max_eff[j] = suffix_max_eff[j + 1] + max(C_list[i][j] for i in feasible_employees_for_task[j])

    suffix_min_res = [0.0] * (m + 1)
    if objective == 2:
        for j in range(m - 1, -1, -1):
            suffix_min_res[j] = suffix_min_res[j + 1] + min(R_list[i][j] for i in feasible_employees_for_task[j])

    def is_better(candidate: float, incumbent: float) -> bool:
        return candidate > incumbent if objective == 1 else candidate < incumbent

    def can_still_assign_remaining_tasks(next_task: int) -> bool:
        for jj in range(next_task, m):
            if not any(used_resources[ii] + R_list[ii][jj] <= b_list[ii] + 1e-9 for ii in feasible_employees_for_task[jj]):
                return False
        return True

    def bound_allows_search(task_index: int, current_eff: float, current_res: float) -> bool:
        if objective == 1:
            return current_eff + suffix_max_eff[task_index] > best_value
        if objective == 2:
            return current_res + suffix_min_res[task_index] < best_value
        current_max_load = max(workload) if workload else 0
        return current_max_load < best_value

    def dfs(task_index: int, current_eff: float, current_res: float) -> None:
        nonlocal best_value, best_assignment, nodes_visited

        if require_used:
            remaining_tasks = m - task_index
            employees_without_tasks = sum(1 for load in workload if load == 0)
            if remaining_tasks < employees_without_tasks:
                return

        nodes_visited += 1

        if not bound_allows_search(task_index, current_eff, current_res):
            return
        if not can_still_assign_remaining_tasks(task_index):
            return

        if task_index == m:
            if require_used and any(load == 0 for load in workload):
                return
            candidate_assignment = current_assignment.copy()
            candidate_value = objective_value(candidate_assignment, C_list, R_list, b_list, objective)
            if is_better(candidate_value, best_value):
                best_value = candidate_value
                best_assignment = candidate_assignment
            return

        j = task_index
        employees = feasible_employees_for_task[j]
        if objective == 3:
            employees = sorted(employees, key=lambda i: (workload[i], used_resources[i] + R_list[i][j], -C_list[i][j]))

        for i in employees:
            new_resource = used_resources[i] + R_list[i][j]
            if new_resource > b_list[i] + 1e-9:
                continue

            current_assignment[j] = i
            used_resources[i] += R_list[i][j]
            workload[i] += 1

            dfs(task_index + 1, current_eff + C_list[i][j], current_res + R_list[i][j])

            workload[i] -= 1
            used_resources[i] -= R_list[i][j]
            current_assignment[j] = -1

    dfs(task_index=0, current_eff=0.0, current_res=0.0)
    elapsed = time.time() - start_time

    if best_assignment is None:
        raise ValueError("Допустимое назначение не найдено. Проверьте ресурсные ограничения b.")

    total_eff, total_res, max_load, final_used_resources, final_workload = evaluate_assignment(best_assignment, C_list, R_list, b_list)
    return AssignmentResult(
        objective_type=objective,
        objective_name=OBJECTIVE_NAMES[objective],
        best_value=float(best_value),
        assignment=best_assignment,
        total_efficiency=float(total_eff),
        total_resources=float(total_res),
        max_workload=int(max_load),
        used_resources_by_employee=final_used_resources,
        workload_by_employee=final_workload,
        elapsed_seconds=float(elapsed),
        nodes_visited=int(nodes_visited),
    )


def solve_all_objectives(
    C: Sequence[Sequence[float]],
    R: Sequence[Sequence[float]],
    b: Sequence[float],
    require_each_employee_used: Optional[bool] = None,
) -> Dict[int, AssignmentResult]:
    """Solves all three single-objective B&B formulations used as ideal points."""
    return {
        objective: branch_and_bound(C, R, b, objective, require_each_employee_used=require_each_employee_used)
        for objective in (1, 2, 3)
    }


def results_to_dataframe(results: Iterable[AssignmentResult], file_name: str = "", sheet_name: Union[int, str] = 0) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append({
            "method": "BB",
            "file_name": file_name,
            "sheet_name": str(sheet_name),
            "objective_type": result.objective_type,
            "objective_name": result.objective_name,
            "best_value": result.best_value,
            "total_efficiency": result.total_efficiency,
            "total_resources": result.total_resources,
            "max_workload": result.max_workload,
            "assignment": str(result.assignment),
            "used_resources_by_employee": str(result.used_resources_by_employee),
            "workload_by_employee": str(result.workload_by_employee),
            "elapsed_seconds": result.elapsed_seconds,
            "nodes_visited": result.nodes_visited,
        })
    return pd.DataFrame(rows)


def ideal_points_dataframe(results: Dict[int, AssignmentResult], file_name: str = "", sheet_name: Union[int, str] = 0) -> pd.DataFrame:
    f1 = results.get(1)
    f2 = results.get(2)
    f3 = results.get(3)
    if f1 is None or f2 is None or f3 is None:
        return pd.DataFrame()
    return pd.DataFrame([{
        "method": "BB",
        "file_name": file_name,
        "sheet_name": str(sheet_name),
        "ideal_f1_efficiency": f1.total_efficiency,
        "ideal_resources_min": f2.total_resources,
        "ideal_workload_min": f3.max_workload,
        "ideal_f2_signed": -f2.total_resources,
        "ideal_f3_signed": -f3.max_workload,
        "bb_ideal_time_sec": f1.elapsed_seconds + f2.elapsed_seconds + f3.elapsed_seconds,
        "bb_nodes_total": f1.nodes_visited + f2.nodes_visited + f3.nodes_visited,
    }])


def save_results_to_excel(
    results: Union[AssignmentResult, Dict[int, AssignmentResult], Sequence[AssignmentResult]],
    output_path: str,
    file_name: str = "",
    sheet_name: Union[int, str] = 0,
) -> str:
    """Saves B&B output in a report format compatible with GA analysis files."""
    if isinstance(results, AssignmentResult):
        result_list = [results]
        result_dict = {results.objective_type: results}
    elif isinstance(results, dict):
        result_dict = dict(results)
        result_list = [result_dict[key] for key in sorted(result_dict)]
    else:
        result_list = list(results)
        result_dict = {result.objective_type: result for result in result_list}

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(p, engine="openpyxl") as writer:
        results_to_dataframe(result_list, file_name=file_name, sheet_name=sheet_name).to_excel(writer, sheet_name="bb_results", index=False)
        ideal_points_dataframe(result_dict, file_name=file_name, sheet_name=sheet_name).to_excel(writer, sheet_name="ideal_points", index=False)
        for result in result_list:
            pd.DataFrame({
                "task_number": list(range(1, len(result.assignment) + 1)),
                "employee_number": [employee + 1 for employee in result.assignment],
                "employee_index_zero_based": result.assignment,
            }).to_excel(writer, sheet_name=f"assignment_f{result.objective_type}", index=False)
    return str(p)


def print_result(result: AssignmentResult) -> None:
    print("=" * 70)
    print("РЕЗУЛЬТАТ МЕТОДА ВЕТВЕЙ И ГРАНИЦ")
    print("=" * 70)
    print(f"Постановка задачи: {result.objective_name}")
    print(f"Значение выбранной целевой функции: {result.best_value}")
    print()
    print("Итоговые значения всех критериев:")
    print(f"  Суммарная эффективность: {result.total_efficiency}")
    print(f"  Суммарные ресурсы:       {result.total_resources}")
    print(f"  Максимальная нагрузка:   {result.max_workload}")
    print()
    print("Назначение задач:")
    print("  Формат: задача -> сотрудник")
    for task_idx, employee_idx in enumerate(result.assignment, start=1):
        print(f"  Задача {task_idx} -> Сотрудник {employee_idx + 1}")
    print()
    print("Нагрузка и ресурсы по сотрудникам:")
    for i, (load, res) in enumerate(zip(result.workload_by_employee, result.used_resources_by_employee), start=1):
        print(f"  Сотрудник {i}: задач = {load}, использовано ресурсов = {res}")
    print()
    print(f"Посещено узлов дерева: {result.nodes_visited}")
    print(f"Время выполнения, сек: {result.elapsed_seconds:.6f}")
    print("=" * 70)
