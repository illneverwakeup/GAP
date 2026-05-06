from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from .config import ExperimentConfig, load_config, save_config_snapshot
from .core import branch_and_bound, parse_objective, print_result, save_results_to_excel, solve_all_objectives
from .data_io import create_excel_template, read_assignment_excel, validate_problem


def setup_logging(verbose: bool = True, log_file: Optional[str] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _sheet(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def _apply_cli_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if getattr(args, "excel", None):
        config.input.excel_path = args.excel
    if getattr(args, "sheet", None) is not None:
        config.input.sheet_name = _sheet(args.sheet)
    if getattr(args, "objective", None) is not None:
        config.method.objective = args.objective
    if getattr(args, "output_dir", None):
        config.output.directory = args.output_dir
    if getattr(args, "output_prefix", None):
        config.output.prefix = args.output_prefix
    if getattr(args, "allow_unused_employees", False):
        config.method.require_each_employee_used = False
    if getattr(args, "quiet", False):
        config.output.verbose = False
    return config


def run_from_config(config: ExperimentConfig) -> str:
    problem = read_assignment_excel(
        config.input.excel_path,
        config.input.sheet_name,
        require_each_employee_used=config.method.require_each_employee_used,
    )
    validate_problem(problem, require_each_employee_used=config.method.require_each_employee_used)

    C = problem.c_matrix.tolist()
    R = problem.r_matrix.tolist()
    b = problem.b_list.tolist()

    objective_raw = str(config.method.objective).strip().lower()
    if objective_raw == "all":
        results = solve_all_objectives(C, R, b, require_each_employee_used=config.method.require_each_employee_used)
        for objective in (1, 2, 3):
            print_result(results[objective])
    else:
        objective = parse_objective(config.method.objective)
        result = branch_and_bound(C, R, b, objective, require_each_employee_used=config.method.require_each_employee_used)
        print_result(result)
        results = {objective: result}

    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config.output.prefix}_{problem.num_agents}x{problem.num_tasks}.xlsx"
    if config.output.write_excel:
        save_results_to_excel(results, str(output_path), file_name=Path(config.input.excel_path).name, sheet_name=config.input.sheet_name)
        save_config_snapshot(config, str(output_dir / f"{config.output.prefix}_{config.hash()}_config.yaml"))
        return str(output_path)
    return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gap-bb",
        description="Метод ветвей и границ для многокритериальной задачи назначения GAP.",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Запустить B&B")
    run_p.add_argument("--config", help="Путь к YAML/JSON конфигу")
    run_p.add_argument("--excel", help="Путь к Excel-файлу данных")
    run_p.add_argument("--sheet", default=None, help="Имя или номер листа Excel")
    run_p.add_argument("--objective", default=None, help="1/2/3, efficiency/resources/workload или all")
    run_p.add_argument("--output-dir", help="Папка результатов")
    run_p.add_argument("--output-prefix", help="Префикс итогового Excel")
    run_p.add_argument("--allow-unused-employees", action="store_true", help="Разрешить сотрудников без задач")
    run_p.add_argument("--quiet", action="store_true", help="Меньше сообщений в консоли")

    val_p = sub.add_parser("validate", help="Проверить Excel-файл без запуска метода")
    val_p.add_argument("--excel", required=True, help="Путь к Excel-файлу данных")
    val_p.add_argument("--sheet", default="0", help="Имя или номер листа Excel")
    val_p.add_argument("--allow-unused-employees", action="store_true")

    tmpl_p = sub.add_parser("create-template", help="Создать Excel-шаблон входных данных")
    tmpl_p.add_argument("--output", required=True, help="Куда сохранить шаблон")
    tmpl_p.add_argument("--agents", required=True, type=int, help="Количество сотрудников")
    tmpl_p.add_argument("--tasks", required=True, type=int, help="Количество задач")
    tmpl_p.add_argument("--example-values", action="store_true", help="Заполнить случайными демонстрационными числами")
    tmpl_p.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "create-template":
        setup_logging(True)
        path = create_excel_template(args.output, args.agents, args.tasks, args.example_values, args.seed)
        print(f"Excel-шаблон создан: {path}")
        return 0

    if args.command == "validate":
        setup_logging(True)
        try:
            problem = read_assignment_excel(args.excel, _sheet(args.sheet), require_each_employee_used=not args.allow_unused_employees)
            validate_problem(problem, require_each_employee_used=not args.allow_unused_employees)
            print("✓ Файл корректен")
            print(f"✓ Количество сотрудников: {problem.num_agents}")
            print(f"✓ Количество задач:       {problem.num_tasks}")
            print(f"✓ Размерность:            {problem.dimension}")
            return 0
        except Exception as exc:
            print("Ошибка проверки Excel:")
            print(exc)
            return 2

    if args.command == "run":
        config = load_config(args.config)
        config = _apply_cli_overrides(config, args)
        log_file = None
        if config.output.write_log:
            log_file = str(Path(config.output.directory) / f"{config.output.prefix}_{config.hash()}.log")
        setup_logging(config.output.verbose, log_file)
        try:
            output_path = run_from_config(config)
            if output_path:
                print(f"Результаты сохранены: {output_path}")
            return 0
        except Exception as exc:
            logging.exception("Запуск завершился ошибкой")
            print("Ошибка запуска:")
            print(exc)
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
