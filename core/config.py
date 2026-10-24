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
    convert_concurrency: int = 4


@dataclass
class SeparateConfig:
    enabled: bool = True
    model: str = "bs_roformer_anvuew"
    isolate_stem: str = "vocals"
    unblend_path: str = "unblend"
    custom_kernels: bool = False
    save_vocals_copy: bool = False


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


@dataclass
class ShazamConfig:
    enabled: bool = False
    identify_concurrency: int = 4


@dataclass
class LyricsConfig:
    enabled: bool = False
    user_agent: str = "audio-transcriber/1.0"
    timeout: float = 10.0
    use_tags_first: bool = True
    use_shazam_result: bool = True
    try_after_asr: bool = True
    fuzzy_threshold: float = 0.6
    transcribe_if_identified_no_lyrics: bool = True
    # NEW:
    use_embedded_lyrics: bool = True     # read lyrics already stored in file tags first
    save_unrecognized: bool = False      # write unresolved texts to JSONL for LRCLIB submission
    unrecognized_path: str = ""          # "" = <output>/unrecognized_lyrics.jsonl


@dataclass
class LrcConfig:
    language: Optional[str] = "ru"
    align_model: Optional[str] = None
    device: str = "auto"
    audio_source: str = "auto"
    embed_in_place: bool = False


@dataclass
class PipelineConfig:
    input_dir: str = ""
    output_dir: str = "transcribed"
    recursive: bool = False
    keep_intermediate: bool = True
    overwrite: bool = True
    mode: str = "sequential"

    convert: ConvertConfig = field(default_factory=ConvertConfig)
    separate: SeparateConfig = field(default_factory=SeparateConfig)
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    shazam: ShazamConfig = field(default_factory=ShazamConfig)
    lyrics: LyricsConfig = field(default_factory=LyricsConfig)
    lrc: LrcConfig = field(default_factory=LrcConfig)

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
