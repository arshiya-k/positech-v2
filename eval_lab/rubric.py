"""Rubrics are YAML files in config/rubrics. The annotation UI and the agreement
metrics are both derived from the file, so a rubric change is one edit."""
import yaml

from eval_lab.db import CONFIG


class Rubric:
    def __init__(self, raw: dict):
        self.raw = raw
        self.id = raw["id"]
        self.version = raw["version"]
        self.item_kind = raw["item_kind"]
        self.dimensions = raw["dimensions"]
        self.flags = raw.get("flags", [])

    @classmethod
    def load(cls, name: str) -> "Rubric":
        return cls(yaml.safe_load((CONFIG / "rubrics" / f"{name}.yaml").read_text()))

    def dimension(self, key: str) -> dict:
        return next(d for d in self.dimensions if d["key"] == key)

    def is_ordinal(self, key: str) -> bool:
        return self.dimension(key)["type"] == "scale"

    def allowed_values(self, key: str) -> list:
        d = self.dimension(key)
        return list(range(d["min"], d["max"] + 1)) if d["type"] == "scale" else list(d["options"])


    def to_ui(self) -> dict:
        """The subset of the rubric the annotation UI needs."""
        return {"id": self.id, "version": self.version, "dimensions": self.dimensions, "flags": self.flags}
