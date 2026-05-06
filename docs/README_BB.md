# Branch and Bound для GAP

Этот модуль содержит классический метод ветвей и границ для задачи назначения GAP. Он используется двумя способами:

1. как самостоятельный метод `BB`;
2. как источник идеальных точек для `NSGA-II` и `enNSGA-II`.

## Где лежит метод

```text
methods/branch_and_bound/
├── __init__.py
├── cli.py
├── config.py
├── core.py
└── data_io.py
```

## Быстрый запуск

```bash
python run_bb.py run --config configs/branch_and_bound/quick.yaml
```

## Запуск по одной целевой функции

```bash
python run_bb.py run --excel examples/example_5x12.xlsx --objective efficiency
python run_bb.py run --excel examples/example_5x12.xlsx --objective resources
python run_bb.py run --excel examples/example_5x12.xlsx --objective workload
```

## Запуск по всем трём критериям

```bash
python run_bb.py run --excel examples/example_5x12.xlsx --objective all
```

## Проверка Excel

```bash
python run_bb.py validate --excel examples/example_5x12.xlsx
```

## Создание шаблона Excel

```bash
python run_bb.py create-template --output examples/template.xlsx --agents 10 --tasks 50
```

## Использование из кода NSGA-II/enNSGA-II

```python
from methods.branch_and_bound.core import branch_and_bound

result_f1 = branch_and_bound(C, R, b, objective_type=1)
result_f2 = branch_and_bound(C, R, b, objective_type=2)
result_f3 = branch_and_bound(C, R, b, objective_type=3)
```

Для совместимости со старыми скриптами также можно оставить в корне файл `branch_and_bound_assignment_1.py`, который перенаправляет импорт на новый модуль.
