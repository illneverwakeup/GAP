from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import hashlib

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class InputConfig:
    excel_path: str = "GA_data_big_size.xlsx"
    sheet_name: Union[int, str] = 0


@dataclass
class MethodConfig:
    name: str = "NSGAII"
    require_each_employee_used: bool = True
    use_repair: bool = True
    use_penalty: bool = True
    save_only_feasible_results: bool = True
    resource_penalty_multiplier: float = 100.0
    unused_employee_penalty_multiplier: float = 10.0


@dataclass
class IdealPointsConfig:
    use_bb: bool = True
    use_cache: bool = True
    cache_path: str = "ideal_points_cache.xlsx"


@dataclass
class GAConfig:
    pop_sizes: List[int] = field(default_factory=lambda: [100])
    crossover_rates: List[float] = field(default_factory=lambda: [0.8])
    mutation_rates: List[float] = field(default_factory=lambda: [0.02])
    generations: List[int] = field(default_factory=lambda: [200])
    repeats: int = 3
    mutation_indpb: float = 0.02


@dataclass
class RandomConfig:
    seed: Optional[int] = 42
    numpy_seed: Optional[int] = 42


@dataclass
class OutputConfig:
    prefix: str = "GA_NSGAII_analysis_ready"
    directory: str = "results"
    write_log: bool = True
    verbose: bool = True


@dataclass
class ExperimentConfig:
    input: InputConfig = field(default_factory=InputConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    ideal_points: IdealPointsConfig = field(default_factory=IdealPointsConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    random: RandomConfig = field(default_factory=RandomConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        raw = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _merge_dataclass(instance: Any, values: Dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise ValueError(f"Неизвестный параметр конфигурации: {key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: Optional[str] = None) -> ExperimentConfig:
    config = ExperimentConfig()
    if not path:
        return config
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("Для чтения YAML установите зависимость: pip install pyyaml")
        loaded = yaml.safe_load(text) or {}
    elif config_path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        raise ValueError("Поддерживаются только .yaml, .yml и .json конфиги")
    if not isinstance(loaded, dict):
        raise ValueError("Файл конфигурации должен содержать словарь параметров")
    return _merge_dataclass(config, loaded)


def save_config_snapshot(config: ExperimentConfig, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        p.write_text(yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        p.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
