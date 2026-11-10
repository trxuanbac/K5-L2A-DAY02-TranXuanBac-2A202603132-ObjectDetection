#!/usr/bin/env python3
"""Evaluate downloaded Colab checkpoints against eval copies in student ID space.

The Colab run used raw eval IDs. Preserve it as historical evidence and compare
only the saved best/last checkpoints here; this is not a new training run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

from prepare_student_dataset import prepare, STUDENT_NAMES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Extracted day2_colab_outputs directory')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,default=Path('outputs/colab_import/corrected'))
    args=parser.parse_args();root=args.root.resolve();source=args.source.resolve()
    out=(root/args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    if root not in out.parents or out.relative_to(root).parts[0]!='outputs':
        raise ValueError('Output must be inside repo outputs/.')
    out.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(out/'ultralytics_config')
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib_config')
    os.environ['YOLO_AUTOINSTALL']='false'
    import torch
    import ultralytics
    from ultralytics import YOLO
    config=prepare(root)
    candidates=[]
    for name in ['best','last']:
        checkpoint=source/f'yolo26_runs/vehicle_annotation/weights/{name}.pt'
        model=YOLO(str(checkpoint))
        assert model.names==STUDENT_NAMES,(model.names,STUDENT_NAMES)
        saved_epochs=len(model.ckpt.get('train_results',{}).get('epoch',[]))
        print('CHECKPOINT',name,'saved epoch rows',saved_epochs,flush=True)
        metrics=model.val(data=str(config),split='val',device='cpu',workers=0,
                          imgsz=640,batch=16,plots=True,project=str(out/'yolo26_runs'),
                          name=f'vehicle_eval_{name}',exist_ok=True)
        per_class={}
        for i,c in enumerate(metrics.box.ap_class_index.tolist()):
            per_class[STUDENT_NAMES[int(c)]]={'precision':float(metrics.box.p[i]),
                'recall':float(metrics.box.r[i]),'map50':float(metrics.box.ap50[i]),
                'map50_95':float(metrics.box.ap[i])}
        result={'checkpoint_name':name+'.pt','checkpoint_sha256':digest(checkpoint),
                'saved_training_history_rows':saved_epochs,
                'checkpoint_epoch':50 if name=='last' else None,
                'checkpoint_local':str(checkpoint),
                'precision':float(metrics.box.mp),'recall':float(metrics.box.mr),
                'map50':float(metrics.box.map50),'map50_95':float(metrics.box.map),
                'per_class':per_class,'eval_run':str(metrics.save_dir)}
        (out/f'metrics_{name}.json').write_text(json.dumps(result,indent=2)+'\n')
        candidates.append(result)
        print('CANDIDATE',json.dumps(result),flush=True)
    selected=max(candidates,key=lambda c:c['map50_95'])
    selected_model=YOLO(selected['checkpoint_local'])
    results=selected_model.predict(source=str(root/'dataset/images/eval'),conf=.25,
        imgsz=640,device='cpu',save=True,save_txt=True,save_conf=True,
        project=str(out/'eval_predictions'),name='visuals',exist_ok=True)
    prediction_dir=Path(results[0].save_dir)
    assert len(list(prediction_dir.glob('*.jpg')))==10
    runtime_manifest=json.loads((root/'outputs/student_training/manifest.json').read_text())
    summary={**selected,'experiment':'student_colab_corrected_eval',
        'evaluation_environment':'local CPU; weights trained in Google Colab',
        'ultralytics':ultralytics.__version__,'torch':torch.__version__,
        'class_names':STUDENT_NAMES,'eval_images':10,'train_images':50,
        'epochs_completed':50,'prediction_dir':str(prediction_dir),
        'eval_raw_to_student_id_map':runtime_manifest['eval_raw_to_student_id_map'],
        'source_eval_labels_unchanged':True,
        'checkpoint_selection':'Highest corrected mAP50-95 among the two saved Colab checkpoints best.pt and last.pt.',
        'selection_limitation':'Intermediate epoch checkpoints were not saved; this does not establish the best corrected score across all 50 epochs.',
        'eval_role':'Validation used for checkpoint selection; not an independent test set.',
        'candidates':candidates}
    (out/'model_eval_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('CORRECTED_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__': main()
