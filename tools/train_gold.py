#!/usr/bin/env python3
"""Train a separate reference experiment using the user's finalized gold labels.

The locked student annotations and their failed QC score remain unchanged.
Gold labels and all intermediate training data stay under ignored local paths.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
GOLD_NAMES = {0: 'car', 1: 'van', 2: 'truck', 3: 'bus'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold-labels', type=Path, default=ROOT/'gold_train_labels')
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--resume-from', type=Path)
    args = parser.parse_args()
    out = ROOT/'outputs/gold_training'
    out.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(out/'ultralytics_config')
    os.environ['MPLCONFIGDIR'] = str(out/'matplotlib_config')
    os.environ['YOLO_AUTOINSTALL'] = 'false'

    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO
    from check_yolo_labels import validate_label_file

    gold = args.gold_labels.resolve()
    train_images = sorted((ROOT/'dataset/images/train').glob('*.jpg'))
    eval_images = sorted((ROOT/'dataset/images/eval').glob('*.jpg'))
    assert len(train_images) == 50 and len(eval_images) == 10
    assert {p.stem for p in gold.glob('*.txt')} == {p.stem for p in train_images}
    assert {p.stem for p in (ROOT/'dataset/labels/eval').glob('*.txt')} == {p.stem for p in eval_images}
    assert not ({digest(p) for p in train_images} & {digest(p) for p in eval_images})
    original_gold = {p.name: digest(p) for p in sorted(gold.glob('*.txt'))}
    original_student = {p.name: digest(p) for p in (ROOT/'dataset/labels/train').glob('*.txt')}
    original_eval = {p.name: digest(p) for p in (ROOT/'dataset/labels/eval').glob('*.txt')}
    dataset = out/'dataset'
    counts = {}
    for split, images, labels in [('train', train_images, gold), ('eval', eval_images, ROOT/'dataset/labels/eval')]:
        image_target, label_target = dataset/'images'/split, dataset/'labels'/split
        image_target.mkdir(parents=True, exist_ok=True)
        label_target.mkdir(parents=True, exist_ok=True)
        histogram = Counter()
        for image in images:
            label = labels/(image.stem+'.txt')
            errors = validate_label_file(label)
            assert not errors, errors
            # Link individual image files. Linking whole directories can cause
            # Ultralytics to resolve back to the original student-label dataset.
            destination = image_target/image.name
            if not destination.exists():
                destination.symlink_to(image.resolve())
            assert destination.resolve() == image.resolve()
            shutil.copy2(label, label_target/label.name)
            assert digest(label_target/label.name) == digest(label)
            histogram.update(int(line.split()[0]) for line in label.read_text().splitlines())
        counts[split] = dict(sorted(histogram.items()))
    config = out/'data_gold.yaml'
    config.write_text(yaml.safe_dump({'path': str(dataset.resolve()), 'train': 'images/train',
                                    'val': 'images/eval', 'names': GOLD_NAMES}, sort_keys=False))
    device = args.device
    if device == 'auto':
        device = '0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    manifest = {
        'label_source': 'finalized gold, explicitly selected by user',
        'experiment': 'gold_reference', 'student_annotation_qc_passed': False,
        'initial_student_commit': '9420f27238129b4ed2a9851db5fac3d55e98471a',
        'source_code_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'script_sha256': digest(Path(__file__)), 'pretrained_weights_sha256': digest(args.weights),
        'class_names': GOLD_NAMES, 'counts_by_raw_id': counts, 'train_images': 50, 'eval_images': 10,
        'gold_sha256': original_gold, 'student_sha256': original_student, 'eval_sha256': original_eval,
        'platform': platform.platform(), 'ultralytics': ultralytics.__version__, 'torch': torch.__version__,
        'device': device, 'epochs_requested': args.epochs, 'batch': args.batch,
        'imgsz': 640, 'seed': 20260909, 'amp': False, 'workers': 0,
        'eval_role': 'validation and checkpoint selection, not an independent test set',
    }
    if args.resume_from:
        previous = json.loads((out/'manifest.json').read_text())
        manifest['initial_run'] = {k: previous[k] for k in ['device','ultralytics','torch','script_sha256']}
        manifest['resume_checkpoint'] = str(args.resume_from)
        manifest['resume_checkpoint_sha256'] = digest(args.resume_from)
        manifest['resume_reason'] = 'MPS progress stalled; resumed the last saved checkpoint on CPU'
    write_json(out/'manifest.json', manifest)
    started = time.monotonic()
    status = {'phase': 'training', 'epoch': 0, 'epochs': args.epochs, 'device': device}
    write_json(out/'status.json', status)
    print('Verified runtime dataset:', counts, 'device:', device, flush=True)
    model = YOLO(str((args.resume_from or args.weights).resolve()))

    def progress(trainer):
        # Ultralytics also fires this callback for the final best-model val,
        # after the training loop has incremented epoch beyond its last index.
        if trainer.epoch >= trainer.epochs:
            return
        status.update(epoch=int(trainer.epoch)+1, elapsed_seconds=round(time.monotonic()-started, 1),
                      metrics={k: float(v) for k,v in trainer.metrics.items()})
        write_json(out/'status.json', status)
        print('GOLD_PROGRESS', json.dumps(status), flush=True)

    model.add_callback('on_fit_epoch_end', progress)
    training_args = dict(data=str(config), epochs=args.epochs, imgsz=640, batch=args.batch, device=device,
                         workers=0, amp=False, seed=20260909, plots=True, cache=False,
                         project=str(out/'yolo26_runs'), name='vehicle_gold', exist_ok=False)
    if args.resume_from:
        training_args['resume'] = True
        training_args['save_dir'] = str(out/'yolo26_runs/vehicle_gold')
    model.train(**training_args)
    best = Path(model.trainer.best)
    assert best.is_file()
    with (Path(model.trainer.save_dir)/'results.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    completed_epochs = max(int(float(row['epoch'])) for row in rows)
    assert completed_epochs == args.epochs
    status['epoch'] = completed_epochs
    status['phase'] = 'validation'
    write_json(out/'status.json', status)
    trained = YOLO(str(best))
    metrics = trained.val(data=str(config), split='val', device=device, workers=0,
                          plots=True, project=str(out/'yolo26_runs'), name='vehicle_gold_eval', exist_ok=False)
    per_class = {}
    for i,c in enumerate(metrics.box.ap_class_index.tolist()):
        per_class[GOLD_NAMES[int(c)]] = {'precision': float(metrics.box.p[i]), 'recall': float(metrics.box.r[i]),
                                       'map50': float(metrics.box.ap50[i]), 'map50_95': float(metrics.box.ap[i])}
    summary = {
        'experiment': 'gold_reference', 'label_source': 'gold, not student annotations',
        'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr),
        'map50': float(metrics.box.map50), 'map50_95': float(metrics.box.map),
        'per_class': per_class, 'class_names': GOLD_NAMES, 'eval_images': 10,
        'checkpoint': str(best), 'train_run': str(model.trainer.save_dir), 'eval_run': str(metrics.save_dir),
        'device': device, 'epochs_completed': status['epoch'], 'ultralytics': ultralytics.__version__,
        'torch': torch.__version__, 'eval_role': manifest['eval_role'],
    }
    status['phase'] = 'prediction'
    write_json(out/'status.json', status)
    predictions = trained.predict(source=str(dataset/'images/eval'), conf=0.25, device=device,
                                  save=True, save_txt=True, save_conf=True,
                                  project=str(out/'eval_predictions'), name='visuals', exist_ok=False)
    prediction_dir = Path(predictions[0].save_dir)
    assert len(list(prediction_dir.glob('*.jpg'))) == 10
    summary['prediction_dir'] = str(prediction_dir)
    assert original_gold == {p.name: digest(p) for p in gold.glob('*.txt')}
    assert original_student == {p.name: digest(p) for p in (ROOT/'dataset/labels/train').glob('*.txt')}
    assert original_eval == {p.name: digest(p) for p in (ROOT/'dataset/labels/eval').glob('*.txt')}
    write_json(out/'model_eval_metrics.json', summary)
    # Public artifacts contain model results, not the gold training labels or
    # train_batch images that would reveal them.
    assets = ROOT/'report/individual/assets'
    for source in [Path(summary['train_run']), Path(summary['eval_run'])]:
        target = assets/'yolo26_runs'/source.name
        target.mkdir(parents=True, exist_ok=True)
        for artifact in source.iterdir():
            if artifact.is_file() and artifact.suffix.lower() in {'.png','.jpg','.csv','.yaml','.json'} and not artifact.name.startswith(('train_batch', 'labels')):
                shutil.copy2(artifact, target/artifact.name)
    target = assets/'eval_predictions'/prediction_dir.name
    target.mkdir(parents=True, exist_ok=True)
    for image in prediction_dir.glob('*.jpg'):
        shutil.copy2(image, target/image.name)
    shutil.copy2(out/'model_eval_metrics.json', assets/'model_eval_metrics_gold.json')
    with zipfile.ZipFile(out/'day2_gold_outputs.zip','w',zipfile.ZIP_DEFLATED) as package:
        package.write(out/'model_eval_metrics.json', 'day2_gold_outputs/model_eval_metrics.json')
        for folder in [out/'yolo26_runs', out/'eval_predictions']:
            for artifact in folder.rglob('*'):
                if artifact.is_file() and not artifact.name.startswith('train_batch'):
                    package.write(artifact, 'day2_gold_outputs/'+artifact.relative_to(out).as_posix())
    status.update(phase='complete', elapsed_seconds=round(time.monotonic()-started, 1), summary=summary)
    write_json(out/'status.json', status)
    print('GOLD_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
