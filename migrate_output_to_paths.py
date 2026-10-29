#!/usr/bin/env python3
"""Migrate flat output/<stem>/ to output/<relpath>/<stem>/ (one-shot)."""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if not input_root.is_dir():
        print(f"input root not found: {input_root}", file=sys.stderr); return 1
    if not output_root.is_dir():
        print(f"output root not found: {output_root}", file=sys.stderr); return 1

    moved = legacy = conflicts = errors = skipped = 0
    artifact_names = (
        "metadata.json", "text.txt", "text.lrc",
        "vocals.wav", "convert.wav",
    )
    for track_dir in sorted(output_root.iterdir()):
        if not track_dir.is_dir():
            continue
        if track_dir.name.startswith("_"):
            continue
        # Never treat an album/parent folder as a flat track folder.
        if not any((track_dir / name).is_file() for name in artifact_names):
            skipped += 1
            continue

        target = output_root / "_legacy" / track_dir.name
        meta_file = track_dir / "metadata.json"
        if meta_file.is_file():
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                stored = data.get("path") if isinstance(data, dict) else None
            except (OSError, ValueError):
                stored = None
            if stored:
                try:
                    rel = Path(stored).relative_to(input_root)
                    target = output_root / rel.with_suffix("")
                except ValueError:
                    target = output_root / "_legacy" / track_dir.name

        if target == track_dir:
            continue
        if track_dir in target.parents:
            print(f"skip: {track_dir.name} -> target inside itself")
            skipped += 1
            continue
        if target.exists():
            print(f"conflict: {track_dir.name} -> "
                  f"{target.relative_to(output_root)}")
            conflicts += 1
            continue

        if args.dry_run:
            print(f"DRY: {track_dir.name} -> {target.relative_to(output_root)}")
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(track_dir), str(target))
            if "_legacy" in target.parts:
                legacy += 1
            else:
                moved += 1
        except OSError as e:
            print(f"error: {track_dir.name}: {e}")
            errors += 1

    print(f"\nmoved: {moved}  quarantined: {legacy}  "
          f"conflicts: {conflicts}  skipped: {skipped}  errors: {errors}  "
          f"(dry_run={args.dry_run})")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
