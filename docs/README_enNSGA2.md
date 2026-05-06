# enNSGA-II для GAP

Папка `methods/ennsga2` содержит пользовательскую версию модифицированного NSGA-II для многокритериальной задачи назначения GAP.

Сохранён исходный концепт метода:

- хромосома задаёт назначение `задача -> сотрудник`;
- оптимизируются три критерия: эффективность, суммарные ресурсы и максимальная нагрузка;
- критерии приведены к максимизации: `f1`, `-resources`, `-max_workload`;
- используется `repair` для возврата решений в допустимую область;
- используется `penalty` для недопустимых решений;
- идеальные точки рассчитываются методом ветвей и границ или берутся из кеша;
- модификация enNSGA-II реализована через локальную min-max нормализацию и отбор по евклидову расстоянию до точки `(1, 1, 1)`;
- результаты сохраняются в Excel с листами `runs_raw`, `ideal_points`, `summary_by_params`, `mann_whitney_ready`, `metadata`, `config`.

## Структура файлов

```text
methods/ennsga2/
├── __init__.py
├── cli.py
├── config.py
├── core.py
├── data_io.py
└── selection.py

configs/ennsga2/
├── quick.yaml
└── paper.yaml

run_ennsga2.py
```

## Быстрый запуск

```bash
pip install -r requirements.txt
python run_ennsga2.py run --config configs/ennsga2/quick.yaml
```

## Проверка Excel без запуска алгоритма

```bash
python run_ennsga2.py validate --excel examples/example_5x12.xlsx
```

## Создание шаблона Excel

```bash
python run_ennsga2.py create-template --output examples/template.xlsx --agents 10 --tasks 50
```

## Запуск с параметрами из командной строки

```bash
python run_ennsga2.py run \
  --excel examples/example_5x12.xlsx \
  --pop-sizes 100 200 \
  --cross 0.8 0.9 \
  --mut 0.02 0.05 \
  --gen 200 \
  --repeats 5
```

## Важное требование

Для расчёта B&B-идеальных точек рядом с `run_ennsga2.py` должен лежать файл:

```text
branch_and_bound_assignment_1.py
```

Если идеальные точки уже есть в `ideal_points_cache.xlsx`, можно отключить расчёт B&B:

```bash
python run_ennsga2.py run --config configs/ennsga2/quick.yaml --no-bb
```

## Где лежит модификация метода

Основная модификация вынесена в файл:

```text
methods/ennsga2/selection.py
```

Класс `EnNSGA2Selector` можно использовать отдельно от полного эксперимента:

```python
from methods.ennsga2.selection import EnNSGA2Selector

selector = EnNSGA2Selector(
    objective_func=calculate_raw_objectives,
    modes=("max", "max", "max"),
)
selected = selector.select_by_ideal_point_distance(population, k=100)
```
