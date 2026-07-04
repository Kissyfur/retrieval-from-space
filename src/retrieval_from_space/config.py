from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _read_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "YAML configs require PyYAML. Install project requirements or use JSON."
            ) from exc
        data = yaml.safe_load(text)
        return {} if data is None else data
    raise ValueError(f"Unsupported config format: {path.suffix}")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@dataclass
class TargetConfig:
    path: str
    target_column: str
    id_column: str = "Id"
    lat_column: str = "lat"
    lon_column: str = "lon"
    time_column: str = "time"
    sheet_name: str | int | None = None
    metadata_columns: list[str] = field(default_factory=list)
    include_spatial_metadata: bool = True
    include_day_metadata: bool = True
    include_cyclic_day_metadata: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TargetConfig":
        return cls(
            path=str(data["path"]),
            target_column=str(data["target_column"]),
            id_column=str(data.get("id_column", "Id")),
            lat_column=str(data.get("lat_column", "lat")),
            lon_column=str(data.get("lon_column", "lon")),
            time_column=str(data.get("time_column", "time")),
            sheet_name=data.get("sheet_name"),
            metadata_columns=list(data.get("metadata_columns", [])),
            include_spatial_metadata=bool(data.get("include_spatial_metadata", True)),
            include_day_metadata=bool(data.get("include_day_metadata", True)),
            include_cyclic_day_metadata=bool(data.get("include_cyclic_day_metadata", True)),
        )


@dataclass
class ProductSpec:
    name: str
    dataset_ids: list[str]
    source: str = "copernicus"
    source_path: str | None = None
    variables: list[str] = field(default_factory=list)
    feature_group: str | None = None
    open_dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    rename_dimensions: dict[str, str] = field(
        default_factory=lambda: {"latitude": "lat", "longitude": "lon"}
    )
    rename_variables: dict[str, str] = field(default_factory=dict)
    matchup: dict[str, Any] = field(default_factory=dict)
    preprocess: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProductSpec":
        dataset_ids = data.get("dataset_ids", data.get("dataset_id"))
        return cls(
            name=str(data["name"]),
            dataset_ids=[str(v) for v in _as_list(dataset_ids)],
            source=str(data.get("source", "copernicus")).lower(),
            source_path=str(data["source_path"]) if data.get("source_path") is not None else None,
            variables=[str(v) for v in data.get("variables", [])],
            feature_group=data.get("feature_group"),
            open_dataset_kwargs=dict(data.get("open_dataset_kwargs", {})),
            rename_dimensions=dict(
                data.get("rename_dimensions", {"latitude": "lat", "longitude": "lon"})
            ),
            rename_variables=dict(data.get("rename_variables", {})),
            matchup=dict(data.get("matchup", {})),
            preprocess=dict(data.get("preprocess", {})),
        )


@dataclass
class MatchupConfig:
    lat_window: float = 0.06
    lon_window: float = 0.06
    time_window_days: int = 1
    lat_threshold: float = 0.1
    lon_threshold: float = 0.1
    time_threshold_days: int = 1
    require_full_time_window: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MatchupConfig":
        data = {} if data is None else data
        return cls(
            lat_window=float(data.get("lat_window", 0.06)),
            lon_window=float(data.get("lon_window", 0.06)),
            time_window_days=int(data.get("time_window_days", 1)),
            lat_threshold=float(data.get("lat_threshold", 0.1)),
            lon_threshold=float(data.get("lon_threshold", 0.1)),
            time_threshold_days=int(data.get("time_threshold_days", 1)),
            require_full_time_window=bool(data.get("require_full_time_window", False)),
        )


@dataclass
class PreprocessConfig:
    positive_quantile: float | None = 0.01
    log_products: bool = True
    add_cloud_land_masks: bool = True
    fillna: float | None = 0.0
    min_valid_ratio: float | None = None
    time_limit: int | None = None
    prefix_variables: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PreprocessConfig":
        data = {} if data is None else data
        return cls(
            positive_quantile=data.get("positive_quantile", 0.01),
            log_products=bool(data.get("log_products", True)),
            add_cloud_land_masks=bool(data.get("add_cloud_land_masks", True)),
            fillna=data.get("fillna", 0.0),
            min_valid_ratio=data.get("min_valid_ratio"),
            time_limit=data.get("time_limit"),
            prefix_variables=bool(data.get("prefix_variables", False)),
        )


@dataclass
class ProblemConfig:
    type: str | None = None
    target_transform: str = "none"
    target_transform_offset: float = 0.0
    class_intervals: list[list[float]] = field(default_factory=list)
    class_labels: list[str] = field(default_factory=list)
    class_encoding: str = "hard"
    soft_label_temperature: float = 1.0
    soft_label_prior: float = 1.0
    test_size: float = 0.2
    random_state: int = 42

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProblemConfig":
        data = {} if data is None else data
        problem_type = data.get("type")
        if problem_type is not None:
            problem_type = str(problem_type).lower()
            if problem_type not in {"classification", "regression"}:
                raise ValueError("problem.type must be 'classification' or 'regression'.")
        class_encoding = str(data.get("class_encoding", "hard")).lower().replace("-", "_")
        aliases = {
            "classes": "hard",
            "hard_classes": "hard",
            "onehot": "one_hot",
            "soft": "soft_probabilities",
            "smooth": "soft_probabilities",
            "smooth_probabilities": "soft_probabilities",
        }
        class_encoding = aliases.get(class_encoding, class_encoding)
        if class_encoding not in {"hard", "one_hot", "soft_probabilities"}:
            raise ValueError(
                "problem.class_encoding must be 'hard', 'one_hot', or 'soft_probabilities'."
            )
        return cls(
            type=problem_type,
            target_transform=str(data.get("target_transform", "none")).lower(),
            target_transform_offset=float(data.get("target_transform_offset", 0.0)),
            class_intervals=[list(map(float, v)) for v in data.get("class_intervals", [])],
            class_labels=[str(v) for v in data.get("class_labels", [])],
            class_encoding=class_encoding,
            soft_label_temperature=float(data.get("soft_label_temperature", 1.0)),
            soft_label_prior=float(data.get("soft_label_prior", 1.0)),
            test_size=float(data.get("test_size", 0.2)),
            random_state=int(data.get("random_state", 42)),
        )


@dataclass
class HyperparameterSearchConfig:
    enabled: bool = False
    cv: int = 3
    scoring: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    param_grid: dict[str, list[Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HyperparameterSearchConfig":
        data = {} if data is None else data
        return cls(
            enabled=bool(data.get("enabled", False)),
            cv=int(data.get("cv", 3)),
            scoring=data.get("scoring"),
            candidates=[dict(v) for v in data.get("candidates", [])],
            param_grid=dict(data.get("param_grid", {})),
        )


@dataclass
class ModelStageConfig:
    family: str = "random_forest"
    feature_groups: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    standardize: bool = False
    sample_weight: Any = None
    augmentation: dict[str, Any] = field(default_factory=dict)
    decision_thresholds: dict[str, Any] = field(default_factory=dict)
    input_selection: dict[str, Any] = field(default_factory=dict)
    hyperparameter_search: HyperparameterSearchConfig = field(default_factory=HyperparameterSearchConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ModelStageConfig":
        data = {} if data is None else data
        return cls(
            family=str(data.get("family", "random_forest")).lower(),
            feature_groups=[str(v) for v in data.get("feature_groups", [])],
            params=dict(data.get("params", {})),
            standardize=bool(data.get("standardize", False)),
            sample_weight=data.get("sample_weight", data.get("make_sample_weight")),
            augmentation=dict(data.get("augmentation", {})),
            decision_thresholds=dict(data.get("decision_thresholds", {})),
            input_selection=dict(data.get("input_selection", {})),
            hyperparameter_search=HyperparameterSearchConfig.from_dict(data.get("hyperparameter_search")),
        )


def _model_stage_mapping_from_dict(data: Any) -> dict[str, ModelStageConfig]:
    if not data:
        return {}
    if isinstance(data, dict):
        return {str(name): ModelStageConfig.from_dict(stage) for name, stage in data.items()}
    if isinstance(data, list):
        stages = {}
        for index, raw_stage in enumerate(data):
            stage_data = dict(raw_stage)
            name = str(stage_data.pop("name", f"base_{index + 1}"))
            stages[name] = ModelStageConfig.from_dict(stage_data)
        return stages
    raise ValueError("model.base_models must be a mapping or a list of named stages.")


@dataclass
class ModelConfig(ModelStageConfig):
    strategy: str = "direct"
    include_base_prediction: bool = True
    base_model: ModelStageConfig | None = None
    base_models: dict[str, ModelStageConfig] = field(default_factory=dict)
    final_model: ModelStageConfig | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ModelConfig":
        data = {} if data is None else data
        strategy = str(data.get("strategy", "direct")).lower()
        if strategy not in {"direct", "stacking", "residual_correction"}:
            raise ValueError("model.strategy must be 'direct', 'stacking', or 'residual_correction'.")
        return cls(
            family=str(data.get("family", "random_forest")).lower(),
            feature_groups=[str(v) for v in data.get("feature_groups", [])],
            params=dict(data.get("params", {})),
            standardize=bool(data.get("standardize", False)),
            sample_weight=data.get("sample_weight", data.get("make_sample_weight")),
            augmentation=dict(data.get("augmentation", {})),
            decision_thresholds=dict(data.get("decision_thresholds", {})),
            input_selection=dict(data.get("input_selection", {})),
            hyperparameter_search=HyperparameterSearchConfig.from_dict(data.get("hyperparameter_search")),
            strategy=strategy,
            include_base_prediction=bool(data.get("include_base_prediction", True)),
            base_model=ModelStageConfig.from_dict(data["base_model"]) if data.get("base_model") else None,
            base_models=_model_stage_mapping_from_dict(data.get("base_models")),
            final_model=ModelStageConfig.from_dict(data["final_model"]) if data.get("final_model") else None,
        )


@dataclass
class PipelineConfig:
    target: TargetConfig
    products: list[ProductSpec]
    output_root: str = "outputs/runs"
    run_id: str | None = None
    run_name: str | None = None
    run_version: str = "v0"
    matchup: MatchupConfig = field(default_factory=MatchupConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        if "target" not in data:
            raise ValueError("Config must contain a 'target' section.")
        if "products" not in data or not data["products"]:
            raise ValueError("Config must contain at least one Copernicus product.")
        return cls(
            target=TargetConfig.from_dict(data["target"]),
            products=[ProductSpec.from_dict(p) for p in data["products"]],
            output_root=str(data.get("output_root", "outputs/runs")),
            run_id=data.get("run_id"),
            run_name=data.get("run_name"),
            run_version=str(data.get("run_version", "v0")),
            matchup=MatchupConfig.from_dict(data.get("matchup")),
            preprocess=PreprocessConfig.from_dict(data.get("preprocess")),
            problem=ProblemConfig.from_dict(data.get("problem")),
            model=ModelConfig.from_dict(data.get("model")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    return PipelineConfig.from_dict(_read_mapping(path))


def write_config_snapshot(config: PipelineConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
