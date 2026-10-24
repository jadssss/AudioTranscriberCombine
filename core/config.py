"""Pipeline configuration dataclasses with JSON save/load."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional


MODES = ("sequential", "staged")


@dataclass
class ConvertConfig:
    enabled: bool = True
    sample_rate: int = 44100
    channels: int = 2
    ffmpeg_path: str = "ffmpeg"


@dataclass
class SeparateConfig:
    enabled: bool = True
    model: str = "bs_roformer_anvuew"
    isolate_stem: str = "vocals"
    unblend_path: str = "unblend"
    custom_kernels: bool = False


@dataclass
class TranscribeConfig:
    enabled: bool = True
    engine: str = "whisperx"
    model: str = "large-v3"
    language: Optional[str] = "ru"
    device: str = "auto"
    compute_type: Optional[str] = None
    batch_size: int = 8
    gigaam_provider: str = "auto"
    local_files_only: bool = False
    vad_filter: bool = True


@dataclass
class PipelineConfig:
    input_dir: str = ""
    output_dir: str = "transcribed"
    recursive: bool = False
    keep_intermediate: bool = True
    overwrite: bool = True
    mode: str = "sequential"           # "sequential" | "staged"

    convert: ConvertConfig = field(default_factory=ConvertConfig)
    separate: SeparateConfig = field(default_factory=SeparateConfig)
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        cfg = cls()
        for top in fields(cls):
            name = top.name
            if name not in data:
                continue
            value = data[name]
            current = getattr(cfg, name)
            if hasattr(current, "__dataclass_fields__"):
                sub = {f.name for f in fields(current)}
                for k, v in value.items():
                    if k in sub:
                        setattr(current, k, v)
            else:
                setattr(cfg, name, value)
        return cfg

    @classmethod
    def from_json(cls, text: str) -> "PipelineConfig":
        return cls.from_dict(json.loads(text))

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
