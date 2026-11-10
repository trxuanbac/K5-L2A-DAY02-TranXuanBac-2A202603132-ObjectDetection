"""Check QC fingerprints and isolate eval normalization from original labels."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/'tools/prepare_student_dataset.py'
spec=importlib.util.spec_from_file_location('prepare_student_dataset',SCRIPT)
prep=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class DatasetPreparationTests(unittest.TestCase):
    def make_repo(self,root):
        for split,count,start in [('train',50,1),('eval',10,51)]:
            images=root/f'dataset/images/{split}';labels=root/f'dataset/labels/{split}'
            images.mkdir(parents=True);labels.mkdir(parents=True)
            for i in range(start,start+count):
                (images/f'drive_{i:03}.jpg').write_bytes(b'image fixture')
                (labels/f'drive_{i:03}.txt').write_text('1 0.5 0.5 0.2 0.3\n')
        files=sorted((root/'dataset/labels/train').glob('*.txt'))
        (root/'outputs').mkdir()
        qc={'passed':True,'f1':.9,'mean_iou_of_matches':.8,'iou_match_threshold':.5,
            'per_image':[{'image':p.stem} for p in files],
            'student_labels_sha256':{p.name:prep.sha256(p) for p in files}}
        (root/'outputs/annotation_evaluation.json').write_text(json.dumps(qc))

    def test_eval_id_changes_only_in_runtime_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_repo(root)
            original=root/'dataset/labels/eval/drive_051.txt'
            before=original.read_bytes()
            config=prep.prepare(root)
            self.assertTrue(config.is_file())
            self.assertEqual(original.read_bytes(),before)
            self.assertEqual((root/'outputs/student_training/dataset/labels/eval/drive_051.txt').read_text(),'3 0.5 0.5 0.2 0.3\n')
            self.assertEqual((root/'outputs/student_training/dataset/labels/train/drive_001.txt').read_text(),'1 0.5 0.5 0.2 0.3\n')

    def test_modified_train_labels_require_fresh_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_repo(root)
            (root/'dataset/labels/train/drive_001.txt').write_text('0 0.5 0.5 0.2 0.3\n')
            with self.assertRaisesRegex(ValueError,'khác phiên'):
                prep.prepare(root)
            self.assertFalse((root/'outputs/student_training').exists())

    def test_runtime_cannot_overwrite_source_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_repo(root)
            with self.assertRaisesRegex(ValueError,'outputs'):
                prep.prepare(root,root/'dataset')


if __name__=='__main__': unittest.main()
