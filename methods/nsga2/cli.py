from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from .config import load_config, ExperimentConfig
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


def _apply_cli_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if getattr(args, "excel", None):
        config.input.excel_path = args.excel
    if getattr(args, "sheet", None) is not None:
        try:
            config.input.sheet_name = int(args.sheet)
        except ValueError:
            config.input.sheet_name = args.sheet
    if getattr(args, "pop_sizes", None):
        config.ga.pop_sizes = args.pop_sizes
    if getattr(args, "cross", None):
        config.ga.crossover_rates = args.cross
    if getattr(args, "mut", None):
        config.ga.mutation_rates = args.mut
    if getattr(args, "gen", None):
        config.ga.generations = args.gen
    if getattr(args, "repeats", None) is not None:
        config.ga.repeats = args.repeats
    if getattr(args, "seed", None) is not None:
        config.random.seed = args.seed
        config.random.numpy_seed = args.seed
    if getattr(args, "output_dir", None):
        config.output.directory = args.output_dir
    if getattr(args, "output_prefix", None):
        config.output.prefix = args.output_prefix
    if getattr(args, "no_bb", False):
        config.ideal_points.use_bb = False
    if getattr(args, "bb", False):
        config.ideal_points.use_bb = True
    if getattr(args, "cache", None):
        config.ideal_points.cache_path = args.cache
    if getattr(args, "allow_unused_employees", False):
        config.method.require_each_employee_used = False
    if getattr(args, "no_repair", False):
        config.method.use_repair = False
    if getattr(args, "no_penalty", False):
        config.method.use_penalty = False
    if getattr(args, "save_all", False):
        config.method.save_only_feasible_results = False
    if getattr(args, "quiet", False):
        config.output.verbose = False
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gap-nsga2",
        description="Удобный запуск NSGA-II для многокритериальной задачи назначения GAP.",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Запустить эксперимент NSGA-II")
    run_p.add_argument("--config", help="Путь к YAML/JSON конфигу")
    run_p.add_argument("--excel", help="Путь к Excel-файлу данных")
    run_p.add_argument("--sheet", help="Имя или номер листа Excel")
    run_p.add_argument("--pop-sizes", nargs="+", type=int, help="Размеры популяции")
    run_p.add_argument("--cross", nargs="+", type=float, help="Вероятности кроссовера")
    run_p.add_argument("--mut", nargs="+", type=float, help="Вероятности мутации")
    run_p.add_argument("--gen", nargs="+", type=int, help="Числа поколений")
    run_p.add_argument("--repeats", type=int, help="Число повторов")
    run_p.add_argument("--seed", type=int, help="Seed для random и numpy")
    run_p.add_argument("--output-dir", help="Папка результатов")
    run_p.add_argument("--output-prefix", help="Префикс итогового Excel")
    run_p.add_argument("--cache", help="Путь к кешу идеальных точек")
    run_p.add_argument("--bb", action="store_true", help="Принудительно включить расчёт B&B")
    run_p.add_argument("--no-bb", action="store_true", help="Отключить расчёт B&B, использовать кеш")
    run_p.add_argument("--allow-unused-employees", action="store_true", help="Разрешить сотрудников без задач")
    run_p.add_argument("--no-repair", action="store_true", help="Отключить repair")
    run_p.add_argument("--no-penalty", action="store_true", help="Отключить penalty")
    run_p.add_argument("--save-all", action="store_true", help="Сохранять также недопустимые решения")
    run_p.add_argument("--quiet", action="store_true", help="Меньше сообщений в консоли")

    val_p = sub.add_parser("validate", help="Проверить Excel-файл без запуска алгоритма")
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
            try:
                sheet = int(args.sheet)
            except ValueError:
                sheet = args.sheet
            problem = read_assignment_excel(
                args.excel,
                sheet,
                require_each_employee_used=not args.allow_unused_employees,
            )
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
        setup_logging(verbose=config.output.verbose, log_file=log_file)
        try:
            from .core import run_experiment
            output_path = run_experiment(config)
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
