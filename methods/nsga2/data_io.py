from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class ProblemData:
    num_agents: int
    num_tasks: int
    c_matrix: np.ndarray
    r_matrix: np.ndarray
    b_list: np.ndarray

    @property
    def dimension(self) -> int:
        return self.num_agents * self.num_tasks


def validate_problem(problem: ProblemData, require_each_employee_used: bool = True) -> None:
    c_matrix = problem.c_matrix
    r_matrix = problem.r_matrix
    b_list = problem.b_list

    if c_matrix.ndim != 2 or c_matrix.size == 0:
        raise ValueError("Матрица эффективности C пустая или имеет неправильный формат.")

    num_agents_local, num_tasks_local = c_matrix.shape

    if num_agents_local == 0:
        raise ValueError("В матрице эффективности C нет сотрудников.")
    if num_tasks_local == 0:
        raise ValueError("В матрице эффективности C нет задач.")
    if r_matrix.shape != (num_agents_local, num_tasks_local):
        raise ValueError(
            "Матрица ресурсов R должна иметь тот же размер, что и C: "
            f"ожидалось {(num_agents_local, num_tasks_local)}, получено {r_matrix.shape}."
        )
    if b_list.shape[0] != num_agents_local:
        raise ValueError(
            "Длина вектора ресурсных ограничений b должна совпадать с количеством сотрудников: "
            f"ожидалось {num_agents_local}, получено {b_list.shape[0]}."
        )

    if require_each_employee_used and num_tasks_local < num_agents_local:
        raise ValueError(
            "Невозможно назначить каждому сотруднику хотя бы одну задачу: "
            "количество задач меньше количества сотрудников."
        )

    if np.isnan(c_matrix).any() or np.isnan(r_matrix).any() or np.isnan(b_list).any():
        raise ValueError("Во входных данных обнаружены пустые или некорректные числовые значения.")

    if np.any(r_matrix < 0) or np.any(b_list < 0):
        raise ValueError("Ресурсы R и ограничения b не должны быть отрицательными.")

    for task_idx in range(num_tasks_local):
        if not np.any(r_matrix[:, task_idx] <= b_list + 1e-9):
            raise ValueError(
                f"Задачу {task_idx + 1} невозможно назначить ни одному сотруднику "
                "без нарушения ресурсного ограничения."
            )


def read_assignment_excel(
    path: str,
    sheet_name: Union[int, str] = 0,
    require_each_employee_used: bool = True,
) -> ProblemData:
    """
    Ожидаемый формат Excel:
    A1 — количество сотрудников n;
    A3 — количество задач m;
    A5:... — матрица эффективности C, размер n x m;
    далее пустая строка;
    далее матрица ресурсов R, размер n x m;
    далее пустая строка;
    далее вектор ресурсных ограничений b, размер n.
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Excel-файл не найден: {path}") from exc
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать Excel-файл '{path}': {exc}") from exc

    try:
        num_agents = int(df.iloc[0, 0])
        num_tasks = int(df.iloc[2, 0])
    except Exception as exc:
        raise ValueError(
            "Не удалось прочитать размерность задачи: "
            "в A1 должно быть число сотрудников, в A3 — число задач."
        ) from exc

    c_start_row = 4
    r_start_row = c_start_row + num_agents + 1
    b_start_row = r_start_row + num_agents + 1

    try:
        c_matrix = df.iloc[c_start_row:c_start_row + num_agents, 0:num_tasks].astype(float).to_numpy()
        r_matrix = df.iloc[r_start_row:r_start_row + num_agents, 0:num_tasks].astype(float).to_numpy()
        b_list = df.iloc[b_start_row:b_start_row + num_agents, 0].astype(float).to_numpy()
    except Exception as exc:
        raise ValueError(
            "Не удалось извлечь C, R и b из Excel. Проверьте, что матрицы и вектор "
            "расположены в ожидаемых строках и содержат только числовые значения."
        ) from exc

    problem = ProblemData(num_agents, num_tasks, c_matrix, r_matrix, b_list)
    validate_problem(problem, require_each_employee_used=require_each_employee_used)
    return problem


def create_excel_template(path: str, agents: int, tasks: int, with_example_values: bool = False, seed: int = 42) -> str:
    if agents <= 0 or tasks <= 0:
        raise ValueError("Количество сотрудников и задач должно быть положительным.")

    rng = np.random.default_rng(seed)
    rows = 4 + agents + 1 + agents + 1 + agents
    cols = max(tasks, 2)
    data = [[None for _ in range(cols)] for _ in range(rows)]
    data[0][0] = agents
    data[2][0] = tasks

    c_start = 4
    r_start = c_start + agents + 1
    b_start = r_start + agents + 1

    if with_example_values:
        c = rng.integers(10, 100, size=(agents, tasks))
        r = rng.integers(1, 20, size=(agents, tasks))
        b = np.maximum(np.ceil(r.sum(axis=1) / max(2, agents)).astype(int), r.max(axis=1))
    else:
        c = np.zeros((agents, tasks), dtype=float)
        r = np.zeros((agents, tasks), dtype=float)
        b = np.zeros(agents, dtype=float)

    for i in range(agents):
        for j in range(tasks):
            data[c_start + i][j] = float(c[i, j])
            data[r_start + i][j] = float(r[i, j])
        data[b_start + i][0] = float(b[i])

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_excel(p, header=False, index=False)
    return str(p)
