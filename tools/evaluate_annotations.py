#!/usr/bin/env python3
"""Class-aware, one-to-one IoU evaluation of student YOLO annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def corners(self) -> tuple[float, float, float, float]:
        return (
            self.x_center - self.width / 2,
            self.y_center - self.height / 2,
            self.x_center + self.width / 2,
            self.y_center + self.height / 2,
        )


def parse_yolo(path: Path) -> list[Box]:
    if not path.exists():
        return []
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{path}:{line_number} phải có 5 cột YOLO")
        class_id = int(values[0])
        x_center, y_center, width, height = map(float, values[1:])
        boxes.append(Box(class_id, x_center, y_center, width, height))
    return boxes


def parse_class_map(value: str) -> dict[int, int]:
    """Parse an explicit gold-ID -> student-ID bijection; never edit gold files."""
    try:
        pairs = [tuple(map(int, item.split(':'))) for item in value.split(',')]
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError
        result = dict(pairs)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('Dùng dạng gold_id:student_id, ví dụ 0:0,1:3,2:1,3:2') from None
    if len(pairs) != 4 or set(result) != {0, 1, 2, 3} or set(result.values()) != {0, 1, 2, 3}:
        raise argparse.ArgumentTypeError('Ánh xạ phải là song ánh đủ bốn ID 0,1,2,3')
    return result


def remap_boxes(boxes: list[Box], class_map: dict[int, int]) -> list[Box]:
    return [Box(class_map[b.class_id], b.x_center, b.y_center, b.width, b.height) for b in boxes]


def iou(first: Box, second: Box) -> float:
    ax1, ay1, ax2, ay2 = first.corners
    bx1, by1, bx2, by2 = second.corners
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union else 0.0


def match(student: list[Box], gold: list[Box], threshold: float) -> list[float]:
    candidates = sorted(
        ((iou(prediction, truth), prediction_index, truth_index)
         for prediction_index, prediction in enumerate(student)
         for truth_index, truth in enumerate(gold)
         if prediction.class_id == truth.class_id),
        reverse=True,
    )
    used_student, used_gold, matches = set(), set(), []
    for overlap, prediction_index, truth_index in candidates:
        if overlap < threshold:
            break
        if prediction_index not in used_student and truth_index not in used_gold:
            used_student.add(prediction_index)
            used_gold.add(truth_index)
            matches.append(overlap)
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description="Chấm nhãn YOLO với gold labels do giảng viên phát sau khi nộp.")
    parser.add_argument("--student-labels", type=Path, required=True)
    parser.add_argument("--gold-labels", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--min-f1", type=float, default=0.80)
    parser.add_argument("--min-mean-iou", type=float, default=0.70)
    parser.add_argument("--output", type=Path, default=Path("outputs/annotation_evaluation.json"))
    parser.add_argument("--gold-class-map", type=parse_class_map, default=None,
                        help="Ánh xạ gold-ID sang student-ID khi hai nguồn dùng thứ tự lớp khác nhau; không sửa gold")
    args = parser.parse_args()

    gold_files = sorted(args.gold_labels.glob("*.txt"))
    if not gold_files:
        raise SystemExit(f"Không có gold labels trong {args.gold_labels}. Hãy đợi giảng viên phát ground truth.")
    missing = [p.name for p in gold_files if not (args.student_labels / p.name).is_file()]
    unexpected = sorted({p.name for p in args.student_labels.glob('*.txt')} - {p.name for p in gold_files})
    if missing or unexpected:
        raise SystemExit(f"Tên file student/gold không khớp; thiếu={missing[:5]}, thừa={unexpected[:5]}")
    true_positive = false_positive = false_negative = 0
    overlaps = []
    per_image = []
    for gold_file in gold_files:
        gold = parse_yolo(gold_file)
        if args.gold_class_map is not None:
            gold = remap_boxes(gold, args.gold_class_map)
        student = parse_yolo(args.student_labels / gold_file.name)
        matched = match(student, gold, args.iou_threshold)
        true_positive += len(matched)
        false_positive += len(student) - len(matched)
        false_negative += len(gold) - len(matched)
        overlaps.extend(matched)
        per_image.append({"image": gold_file.stem, "gold_boxes": len(gold), "student_boxes": len(student), "matches": len(matched)})

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = sum(overlaps) / len(overlaps) if overlaps else 0.0
    passed = f1 >= args.min_f1 and mean_iou >= args.min_mean_iou
    result = {
        "iou_match_threshold": args.iou_threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou_of_matches": round(mean_iou, 4),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "passed": passed,
        "required_f1": args.min_f1,
        "required_mean_iou": args.min_mean_iou,
        "per_image": per_image,
        "student_labels_sha256": {
            p.name: hashlib.sha256((args.student_labels / p.name).read_bytes()).hexdigest()
            for p in gold_files
        },
        "gold_labels_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in gold_files
        },
    }
    if args.gold_class_map is not None:
        result['gold_class_map'] = args.gold_class_map
        result['class_id_space'] = 'student: 0=car, 1=truck, 2=bus, 3=van'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nĐẠT: có thể sang bước train YOLO." if passed else "\nCHƯA ĐẠT: sửa nhãn theo lỗi FN/FP và box lỏng rồi chạy lại.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
