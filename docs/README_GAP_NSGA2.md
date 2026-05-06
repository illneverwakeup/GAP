# GAP NSGA-II: пользовательская версия скриптов

Эта версия сохраняет концепцию исходного скрипта:

- многокритериальная GAP-постановка;
- хромосома `задача -> сотрудник`;
- классический NSGA-II из DEAP;
- критерии: эффективность, минус суммарные ресурсы, минус максимальная нагрузка;
- `repair` и `penalty` для ограничений;
- расчёт идеальных точек через B&B;
- итоговый Excel с листами `runs_raw`, `ideal_points`, `summary_by_params`, `mann_whitney_ready`.

Добавлено для удобства сторонних пользователей:

- запуск через CLI;
- настройки в YAML/JSON-конфиге;
- проверка Excel без запуска алгоритма;
- создание Excel-шаблона;
- логирование;
- сохранение metadata и полного config в результатный Excel;
- seed для воспроизводимости;
- понятная ошибка, если отсутствует `branch_and_bound_assignment_1.py`.

## Установка

```bash
pip install -r requirements.txt
```

## Быстрый запуск

```bash
python run_nsga2.py run --config configs/nsga2_quick.yaml
```

Или без конфига:

```bash
python run_nsga2.py run --excel GA_data_big_size.xlsx --pop-sizes 100 --cross 0.8 --mut 0.02 --gen 200 --repeats 3
```

## Экспериментальный профиль для статьи

```bash
python run_nsga2.py run --config configs/nsga2_paper.yaml
```

## Проверка Excel

```bash
python run_nsga2.py validate --excel GA_data_big_size.xlsx
```

## Создание шаблона Excel

Пустой шаблон:

```bash
python run_nsga2.py create-template --output examples/template.xlsx --agents 10 --tasks 50
```

Шаблон с демонстрационными числами:

```bash
python run_nsga2.py create-template --output examples/example.xlsx --agents 10 --tasks 50 --example-values
```

## Формат Excel-файла

- `A1` — количество сотрудников `n`;
- `A3` — количество задач `m`;
- `A5` и далее — матрица эффективности `C`, размер `n x m`;
- затем пустая строка;
- затем матрица ресурсов `R`, размер `n x m`;
- затем пустая строка;
- затем вектор ресурсных ограничений `b`, размер `n`.

## Важная зависимость B&B

Если включён расчёт идеальных точек:

```yaml
ideal_points:
  use_bb: true
```

рядом с запускным скриптом должен находиться файл:

```text
branch_and_bound_assignment_1.py
```

Если B&B-файл недоступен, можно использовать заранее подготовленный кеш:

```yaml
ideal_points:
  use_bb: false
  use_cache: true
  cache_path: "ideal_points_cache.xlsx"
```

## Результаты

Результат сохраняется в папку `results/`.

Итоговый Excel содержит листы:

- `runs_raw` — все сохранённые решения;
- `ideal_points` — идеальные точки B&B;
- `summary_by_params` — агрегированная сводка по параметрам;
- `mann_whitney_ready` — таблица для статистического сравнения;
- `metadata` — сведения о запуске, версиях и seed;
- `config` — полный набор параметров запуска.

Также рядом сохраняется snapshot конфига и `.log` файл.
