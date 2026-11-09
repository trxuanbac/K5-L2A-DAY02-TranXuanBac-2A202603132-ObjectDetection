#!/usr/bin/env python3
"""Describe saved gold-reference predictions at conf=0.25, IoU=0.50.

This diagnostic uses public eval labels only. It is not a replacement for the
confidence-swept Ultralytics validation metrics or student annotation QC.
"""
import json
from pathlib import Path

from evaluate_annotations import Box, iou, parse_yolo

ROOT = Path(__file__).resolve().parents[1]
NAMES = {0: 'car', 1: 'van', 2: 'truck', 3: 'bus'}


def main():
    metrics = json.loads((ROOT/'outputs/gold_training/model_eval_metrics.json').read_text())
    predictions = Path(metrics['prediction_dir'])/'labels'
    per_image, wrong_classes, overlaps = [], [], []
    for path in sorted((ROOT/'dataset/labels/eval').glob('*.txt')):
        truths = parse_yolo(path)
        pred_path = predictions/path.name
        lines = pred_path.read_text().splitlines() if pred_path.exists() else []
        preds = [Box(int(p[0]), *map(float, p[1:5])) for p in (line.split() for line in lines)]
        candidates = sorted(((iou(p,t),pi,ti) for pi,p in enumerate(preds) for ti,t in enumerate(truths)
                             if p.class_id == t.class_id), reverse=True)
        used_p, used_t, matched = set(), set(), []
        for overlap, pi, ti in candidates:
            if overlap < 0.50: break
            if pi not in used_p and ti not in used_t:
                used_p.add(pi);used_t.add(ti);matched.append((overlap,pi,ti));overlaps.append(overlap)
        by_class = {}
        for c,name in NAMES.items():
            tp = sum(truths[ti].class_id == c for _,_,ti in matched)
            by_class[name] = {'gold':sum(t.class_id == c for t in truths),
                              'predictions':sum(p.class_id == c for p in preds), 'tp':tp,
                              'fn':sum(t.class_id == c for i,t in enumerate(truths) if i not in used_t),
                              'fp':sum(p.class_id == c for i,p in enumerate(preds) if i not in used_p)}
        for ti,t in enumerate(truths):
            if ti in used_t: continue
            options = [(iou(p,t),pi) for pi,p in enumerate(preds) if pi not in used_p and p.class_id != t.class_id]
            if options:
                overlap,pi = max(options)
                if overlap >= .50:
                    wrong_classes.append({'image':path.stem+'.jpg','expected':NAMES[t.class_id],
                                          'predicted':NAMES[preds[pi].class_id],'iou':round(overlap,4)})
        per_image.append({'image':path.stem+'.jpg','classes':by_class,'tp':len(matched),
                          'fp':len(preds)-len(matched),'fn':len(truths)-len(matched)})
    lowest = min(metrics['per_class'], key=lambda c:metrics['per_class'][c]['recall'])
    examples = [r for r in per_image if r['classes'][lowest]['gold']]
    example = max(examples, key=lambda r:(r['classes'][lowest]['fn'],r['classes'][lowest]['gold']))
    report = {'confidence':.25,'iou_threshold':.50,'class_names':NAMES,
              'note':'Fixed-threshold illustration, distinct from confidence-swept validation metrics.',
              'lowest_validation_recall_class':lowest,
              'lowest_validation_recall':metrics['per_class'][lowest]['recall'],
              'illustration':example,'wrong_class_examples':wrong_classes,'per_image':per_image,
              'matched_boxes':len(overlaps),'matches_iou_below_075':sum(x<.75 for x in overlaps),
              'mean_iou_of_matches':sum(overlaps)/len(overlaps) if overlaps else 0}
    target=ROOT/'report/individual/assets/gold_prediction_analysis.json'
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='per_image'},indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
