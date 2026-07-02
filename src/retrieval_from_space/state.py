from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class PipelineState:
    path: Path
    stages: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineState":
        path = Path(path)
        if not path.exists():
            return cls(path=path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path=path,
            stages=dict(data.get("stages", {})),
            artifacts=dict(data.get("artifacts", {})),
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "stages": self.stages,
            "artifacts": self.artifacts,
        }
        self.path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def mark(self, stage: str, status: str, artifacts: dict[str, Any] | None = None) -> None:
        self.stages[stage] = status
        if artifacts:
            self.artifacts.setdefault(stage, {}).update(artifacts)
        self.save()
