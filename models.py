from dataclasses import dataclass, asdict, field
from typing import List, Any
import json


@dataclass
class Connector:
    id: str
    type: str = "pipe"
    properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Motor:
    name: str
    connectors: List[Connector] = field(default_factory=list)
    properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "connectors": [c.to_dict() for c in self.connectors],
            "properties": self.properties,
        }


@dataclass
class ProcessingUnit:
    name: str
    poll_interval: float = 1.0
    motors: List[Motor] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "poll_interval": self.poll_interval,
            "motors": [m.to_dict() for m in self.motors],
            "metadata": self.metadata,
        }

def export_models(models: List[ProcessingUnit], path: str = "models.json") -> str:
    """Export a list of ProcessingUnit dataclasses to a JSON file and return path."""
    arr = [m.to_dict() for m in models]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(arr, f, indent=2)
    return path
