#!/usr/bin/env python3
"""Mark tracks with metadata.empty_text=true if their text is missing or hallucinated."""
import json, sys
from pathlib import Path

HALLUCINATION = (
    "продолжение следует", "субтитры сделан", "субтитры создава",
    "created with", "russian-area.de", "музыка музыка",
)


def main(root: Path) -> int:
    count = 0
    for meta in root.rglob("metadata.json"):
        track = meta.parent
        txt = track / "text.txt"
        if txt.is_file():
            body = txt.read_text(encoding="utf-8", errors="ignore").strip().lower()
            # If the text is real (not hallucination) and not empty, skip marking.
            if body and not any(h in body for h in HALLUCINATION):
                continue
            # Remove the text.txt so the next run will re-process it.
            txt.unlink()
        # Set empty_text=True in metadata.
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data["empty_text"] = True
        meta.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(f"marked: {track}")
        count += 1
    print(f"\nmarked {count} tracks")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))