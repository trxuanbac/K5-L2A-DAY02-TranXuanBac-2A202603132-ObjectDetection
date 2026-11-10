#!/usr/bin/env python3
"""Build a QC-locked runtime dataset, normalizing only copies of eval labels."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path

STUDENT_NAMES = {0: 'car', 1: 'truck', 2: 'bus', 3: 'van'}
# Supplied gold/eval raw schema: 0=car, 1=van, 2=truck, 3=bus.
EVAL_TO_STUDENT = {0: 0, 1: 3, 2: 1, 3: 2}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root: Path, output: Path | None = None) -> Path:
    root = root.resolve()
    output = (output or root / 'outputs/student_training').resolve()
    if root not in output.parents or output.relative_to(root).parts[0] != 'outputs':
        raise ValueError('Runtime phải nằm trong outputs/ của repo, tách khỏi dữ liệu gốc.')
    qc = json.loads((root / 'outputs/annotation_evaluation.json').read_text())
    train = root / 'dataset/labels/train'
    labels = sorted(train.glob('*.txt'))
    actual_hashes = {p.name: sha256(p) for p in labels}
    if not (qc.get('passed') is True and qc.get('f1', 0) >= .80
            and qc.get('mean_iou_of_matches', 0) >= .70
            and qc.get('iou_match_threshold') == .50):
        raise ValueError('CP5 chưa đạt cả hai ngưỡng theo bài.')
    if len(labels) != 50 or actual_hashes != qc.get('student_labels_sha256'):
        raise ValueError('Nhãn train khác phiên đã được chấm QC; cần chạy lại evaluator.')
    if {p.stem for p in labels} != {p['image'] for p in qc['per_image']}:
        raise ValueError('Kết quả CP5 không bao phủ đúng split train.')
    runtime = output / 'dataset'
    source_eval = root / 'dataset/labels/eval'
    eval_before = {p.name: sha256(p) for p in sorted(source_eval.glob('*.txt'))}
    source_images = {}
    for split, count in [('train', 50), ('eval', 10)]:
        images = sorted(p for p in (root / f'dataset/images/{split}').iterdir()
                        if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        source_labels = train if split == 'train' else source_eval
        if len(images) != count or {p.stem for p in images} != {p.stem for p in source_labels.glob('*.txt')}:
            raise ValueError(f'Split {split} phải có đúng {count} ảnh và nhãn khớp tên.')
        source_images[split] = [p.name for p in images]
        image_dest, label_dest = runtime / f'images/{split}', runtime / f'labels/{split}'
        image_dest.mkdir(parents=True, exist_ok=True)
        label_dest.mkdir(parents=True, exist_ok=True)
        if {p.name for p in image_dest.iterdir()} - {p.name for p in images}:
            raise ValueError(f'Runtime {split} có ảnh thừa; dùng một thư mục outputs mới.')
        if {p.name for p in label_dest.glob('*.txt')} - {p.stem + '.txt' for p in images}:
            raise ValueError(f'Runtime {split} có nhãn thừa; dùng một thư mục outputs mới.')
        for image in images:
            shutil.copy2(image, image_dest / image.name)
            label = source_labels / (image.stem + '.txt')
            if split == 'train':
                shutil.copy2(label, label_dest / label.name)
            else:
                normalized = []
                for line in label.read_text().splitlines():
                    if not line.strip():
                        continue
                    parts = line.split()
                    parts[0] = str(EVAL_TO_STUDENT[int(parts[0])])
                    normalized.append(' '.join(parts))
                (label_dest / label.name).write_text('\n'.join(normalized) + '\n')
    assert not set(source_images['train']) & set(source_images['eval'])
    assert eval_before == {p.name: sha256(p) for p in source_eval.glob('*.txt')}
    data_yaml = output / 'data_student.yaml'
    data_yaml.write_text(f'path: {json.dumps(str(runtime))}\ntrain: images/train\nval: images/eval\nnames:\n'
                         + ''.join(f'  {i}: {name}\n' for i, name in STUDENT_NAMES.items()))
    manifest = {'experiment': 'student_rework', 'student_class_names': STUDENT_NAMES,
                'eval_raw_to_student_id_map': EVAL_TO_STUDENT,
                'train_labels_sha256': actual_hashes, 'source_eval_labels_sha256': eval_before,
                'train_images': source_images['train'], 'eval_images': source_images['eval'],
                'annotation_f1': qc['f1'], 'annotation_mean_iou': qc['mean_iou_of_matches'],
                'source_eval_labels_unchanged': True, 'gold_train_files_used': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return data_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(prepare(args.root, args.output))


if __name__ == '__main__':
    main()
