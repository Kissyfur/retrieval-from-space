from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def matchups(self) -> Path:
        return self.processed / "matchups"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def state_file(self) -> Path:
        return self.root / "pipeline_state.json"

    def ensure(self) -> "RunPaths":
        for path in (
            self.config,
            self.logs,
            self.raw,
            self.processed,
            self.matchups,
            self.datasets,
            self.models,
            self.metrics,
            self.reports,
            self.checkpoints,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def make_run_id(run_name: str | None = None, run_version: str | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix_parts = [part for part in (run_name, run_version) if part]
    if not suffix_parts:
        return stamp
    suffix = "_".join(suffix_parts)
    safe_suffix = "".join(c if c.isalnum() or c in {"-", "_", "."} else "-" for c in suffix)
    return f"{stamp}_{safe_suffix.strip('-')}"


def resolve_run_paths(
    output_root: str | Path,
    run_id: str | None = None,
    run_name: str | None = None,
    run_version: str | None = None,
) -> RunPaths:
    output_root = Path(output_root)
    run_id = run_id or make_run_id(run_name, run_version)
    return RunPaths(output_root / run_id).ensure()
