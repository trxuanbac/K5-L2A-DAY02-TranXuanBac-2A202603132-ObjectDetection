#!/usr/bin/env python3
"""Describe corrected Colab predictions at conf .25 and IoU .50 on eval only."""
import argparse
import json
from pathlib import Path
from evaluate_annotations import Box,iou,parse_yolo


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics',type=Path,required=True)
    parser.add_argument('--eval-labels',type=Path,required=True,help='Eval copies normalized to student IDs')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--example',help='Visually reviewed eval image to illustrate the lowest-recall class')
    args=parser.parse_args();m=json.loads(args.metrics.read_text())
    names={int(i):name for i,name in m['class_names'].items()}
    predictions=Path(m['prediction_dir'])/'labels'
    overlaps=[];per_image=[];wrong=[]
    for path in sorted(args.eval_labels.glob('*.txt')):
        truths=parse_yolo(path)
        pred_path=predictions/path.name
        preds=[Box(int(p[0]),*map(float,p[1:5])) for p in (l.split() for l in pred_path.read_text().splitlines())] if pred_path.exists() else []
        candidates=sorted(((iou(p,t),pi,ti) for pi,p in enumerate(preds) for ti,t in enumerate(truths) if p.class_id==t.class_id),reverse=True)
        used_p=set();used_t=set();matched=[]
        for overlap,pi,ti in candidates:
            if overlap<.5:break
            if pi not in used_p and ti not in used_t:
                used_p.add(pi);used_t.add(ti);matched.append((overlap,pi,ti));overlaps.append(overlap)
        by_class={}
        for c,name in names.items():
            by_class[name]={'truth':sum(t.class_id==c for t in truths),
                'predictions':sum(p.class_id==c for p in preds),
                'tp':sum(truths[ti].class_id==c for _,_,ti in matched),
                'fn':sum(t.class_id==c for i,t in enumerate(truths) if i not in used_t),
                'fp':sum(p.class_id==c for i,p in enumerate(preds) if i not in used_p)}
        unmatched=sorted(((iou(p,t),pi,ti) for pi,p in enumerate(preds) for ti,t in enumerate(truths)
                          if pi not in used_p and ti not in used_t and p.class_id!=t.class_id),reverse=True)
        wrong_p=set();wrong_t=set()
        for overlap,pi,ti in unmatched:
            if overlap<.5:break
            if pi in wrong_p or ti in wrong_t:continue
            wrong_p.add(pi);wrong_t.add(ti)
            wrong.append({'image':path.stem+'.jpg','expected':names[truths[ti].class_id],
                'predicted':names[preds[pi].class_id],'iou':round(overlap,4),
                'truth_xyxy_normalized':list(truths[ti].corners),
                'predicted_xyxy_normalized':list(preds[pi].corners)})
        per_image.append({'image':path.stem+'.jpg','classes':by_class,'tp':len(matched),
                          'fp':len(preds)-len(matched),'fn':len(truths)-len(matched)})
    lowest=min(m['per_class'],key=lambda c:m['per_class'][c]['recall'])
    illustration=max((r for r in per_image if r['classes'][lowest]['truth']),
                     key=lambda r:(r['classes'][lowest]['fn'],r['classes'][lowest]['truth']))
    if args.example:
        illustration=next(r for r in per_image if r['image']==args.example)
        if not illustration['classes'][lowest]['truth']:
            raise ValueError('Example does not contain the lowest-recall class.')
    result={'experiment':m['experiment'],'checkpoint_name':m['checkpoint_name'],
        'checkpoint_sha256':m['checkpoint_sha256'],'class_names':names,
        'confidence':.25,'iou_threshold':.5,
        'note':'Fixed-confidence illustration; not the confidence-swept validation recall or annotation F1.',
        'lowest_validation_recall_class':lowest,'lowest_validation_recall':m['per_class'][lowest]['recall'],
        'illustration':illustration,'wrong_class_examples':wrong,'per_image':per_image,
        'matched_boxes':len(overlaps),'matches_iou_below_075':sum(x<.75 for x in overlaps),
        'mean_iou_of_matches':sum(overlaps)/len(overlaps) if overlaps else 0}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_image'},indent=2))


if __name__=='__main__':main()
