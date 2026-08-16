from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def safe_float(value: str) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def iter_csv_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    yield from sorted(path.glob("*.csv"))


def evaluate_file(path: Path, dist_threshold: float, max_visual_age: float) -> Dict[str, float]:
    total_gt = 0
    tp = 0
    fn = 0
    fp = 0
    errors = []
    yaw_errors = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("source") != "gt":
                continue

            gt_x = safe_float(row.get("gt_x", ""))
            gt_y = safe_float(row.get("gt_y", ""))
            gt_yaw = safe_float(row.get("gt_yaw", ""))
            gt_stamp = safe_float(row.get("gt_stamp", ""))
            vis_x = safe_float(row.get("visual_x", ""))
            vis_y = safe_float(row.get("visual_y", ""))
            vis_yaw = safe_float(row.get("visual_yaw", ""))
            vis_stamp = safe_float(row.get("visual_stamp", ""))

            if gt_x is None or gt_y is None or gt_stamp is None:
                continue
            total_gt += 1

            if vis_x is None or vis_y is None or vis_stamp is None:
                fn += 1
                continue
            if abs(gt_stamp - vis_stamp) > max_visual_age:
                fn += 1
                continue

            err = math.hypot(vis_x - gt_x, vis_y - gt_y)
            if err <= dist_threshold:
                tp += 1
                errors.append(err)
                if gt_yaw is not None and vis_yaw is not None:
                    yaw_errors.append(abs((vis_yaw - gt_yaw + math.pi) % (2.0 * math.pi) - math.pi))
            else:
                fn += 1
                fp += 1

    rmse = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else float("nan")
    mae = sum(errors) / len(errors) if errors else float("nan")
    yaw_mae = sum(yaw_errors) / len(yaw_errors) if yaw_errors else float("nan")
    mota = 1.0 - (fn + fp) / total_gt if total_gt else float("nan")
    coverage = tp / total_gt if total_gt else float("nan")
    return {
        "file": str(path),
        "gt_frames": total_gt,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "coverage": coverage,
        "mota_pose": mota,
        "mae_m": mae,
        "rmse_m": rmse,
        "yaw_mae_rad": yaw_mae,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate tracking CSV logs")
    parser.add_argument("path", help="CSV file or directory with CSV files")
    parser.add_argument("--dist-threshold", type=float, default=0.35)
    parser.add_argument("--max-visual-age", type=float, default=0.25)
    args = parser.parse_args()

    rows: List[Dict[str, float]] = []
    for csv_path in iter_csv_files(Path(args.path).expanduser()):
        rows.append(evaluate_file(csv_path, args.dist_threshold, args.max_visual_age))

    if not rows:
        print("No CSV files found")
        return

    headers = ["file", "gt_frames", "tp", "fn", "fp", "coverage", "mota_pose", "mae_m", "rmse_m", "yaw_mae_rad"]
    print(",".join(headers))
    for row in rows:
        values = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                values.append(f"{v:.6f}")
            else:
                values.append(str(v))
        print(",".join(values))


if __name__ == "__main__":
    main()
