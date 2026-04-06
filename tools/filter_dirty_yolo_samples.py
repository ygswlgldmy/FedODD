#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class DirtySample:
    split: str
    label_path: Path
    image_path: Path | None
    line_no: int
    raw_line: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan a YOLO dataset and isolate samples with invalid bbox labels."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="YOLO dataset root containing images/ and labels/.",
    )
    parser.add_argument(
        "--mode",
        choices=("report", "move", "delete"),
        default="report",
        help="Only report, move bad samples to quarantine, or delete them.",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=None,
        help="Where to move dirty samples when --mode move is used. Defaults to <dataset_root>_dirty.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Tolerance for tiny floating-point overflow. Use 1e-6 if you want to ignore very small drift.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Do not treat empty label files as dirty.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every dirty sample instead of only the summary.",
    )
    return parser.parse_args()


def find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def analyze_label_file(label_path: Path, image_path: Path | None, split: str, eps: float, allow_empty: bool) -> list[DirtySample]:
    dirty: list[DirtySample] = []
    lines = label_path.read_text(encoding="utf-8").splitlines()

    if not lines and not allow_empty:
        dirty.append(
            DirtySample(
                split=split,
                label_path=label_path,
                image_path=image_path,
                line_no=0,
                raw_line="",
                reason="empty label file",
            )
        )
        return dirty

    for idx, raw_line in enumerate(lines, start=1):
        parts = raw_line.split()
        if len(parts) != 5:
            dirty.append(
                DirtySample(
                    split=split,
                    label_path=label_path,
                    image_path=image_path,
                    line_no=idx,
                    raw_line=raw_line,
                    reason=f"expected 5 columns, got {len(parts)}",
                )
            )
            continue

        try:
            _, x_str, y_str, w_str, h_str = parts
            x = float(x_str)
            y = float(y_str)
            w = float(w_str)
            h = float(h_str)
        except ValueError:
            dirty.append(
                DirtySample(
                    split=split,
                    label_path=label_path,
                    image_path=image_path,
                    line_no=idx,
                    raw_line=raw_line,
                    reason="non-numeric bbox value",
                )
            )
            continue

        if w <= 0.0 or h <= 0.0:
            dirty.append(
                DirtySample(
                    split=split,
                    label_path=label_path,
                    image_path=image_path,
                    line_no=idx,
                    raw_line=raw_line,
                    reason=f"non-positive size w={w}, h={h}",
                )
            )
            continue

        x_min = x - w / 2.0
        y_min = y - h / 2.0
        x_max = x + w / 2.0
        y_max = y + h / 2.0

        if x_min < -eps or y_min < -eps or x_max > 1.0 + eps or y_max > 1.0 + eps:
            dirty.append(
                DirtySample(
                    split=split,
                    label_path=label_path,
                    image_path=image_path,
                    line_no=idx,
                    raw_line=raw_line,
                    reason=(
                        f"bbox out of range: "
                        f"xmin={x_min:.9f}, ymin={y_min:.9f}, xmax={x_max:.9f}, ymax={y_max:.9f}"
                    ),
                )
            )

    return dirty


def move_to_quarantine(path: Path, dataset_root: Path, quarantine_root: Path) -> Path:
    relative = path.relative_to(dataset_root)
    destination = quarantine_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    return destination


def delete_if_exists(path: Path | None) -> None:
    if path is not None and path.exists():
        path.unlink()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    labels_root = dataset_root / "labels"
    images_root = dataset_root / "images"

    if not labels_root.exists() or not images_root.exists():
        raise FileNotFoundError(f"{dataset_root} must contain images/ and labels/ directories.")

    quarantine_root = (
        args.quarantine_dir.resolve()
        if args.quarantine_dir is not None
        else dataset_root.parent / f"{dataset_root.name}_dirty"
    )

    dirty_by_label: dict[Path, list[DirtySample]] = {}

    for split_dir in sorted(labels_root.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        images_dir = images_root / split
        for label_path in sorted(split_dir.glob("*.txt")):
            image_path = find_image(images_dir, label_path.stem)
            dirty = analyze_label_file(
                label_path=label_path,
                image_path=image_path,
                split=split,
                eps=args.eps,
                allow_empty=args.allow_empty,
            )
            if image_path is None:
                dirty.append(
                    DirtySample(
                        split=split,
                        label_path=label_path,
                        image_path=None,
                        line_no=0,
                        raw_line="",
                        reason="missing paired image file",
                    )
                )
            if dirty:
                dirty_by_label[label_path] = dirty

    total_dirty_files = len(dirty_by_label)
    total_dirty_lines = sum(len(items) for items in dirty_by_label.values())

    if args.verbose:
        for items in dirty_by_label.values():
            first = items[0]
            print(f"[{first.split}] {first.label_path}")
            for item in items:
                prefix = f"  line {item.line_no}: " if item.line_no else "  file: "
                print(f"{prefix}{item.reason}")
                if item.raw_line:
                    print(f"    {item.raw_line}")

    if args.mode == "move":
        for label_path, items in dirty_by_label.items():
            image_path = items[0].image_path
            moved_label = move_to_quarantine(label_path, dataset_root, quarantine_root)
            if image_path is not None and image_path.exists():
                move_to_quarantine(image_path, dataset_root, quarantine_root)
            print(f"moved {label_path} -> {moved_label}")
    elif args.mode == "delete":
        for label_path, items in dirty_by_label.items():
            image_path = items[0].image_path
            delete_if_exists(label_path)
            delete_if_exists(image_path)
            print(f"deleted {label_path}")

    print(
        f"dirty files: {total_dirty_files}, dirty label entries: {total_dirty_lines}, mode: {args.mode}"
    )
    if total_dirty_files and args.mode == "report":
        print("Use --verbose to print details for every file.")
        print("Use --mode move to quarantine dirty image/label pairs.")
        print("Use --mode delete to remove dirty image/label pairs.")


if __name__ == "__main__":
    main()
