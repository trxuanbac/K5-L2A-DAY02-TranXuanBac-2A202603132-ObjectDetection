"""Protect class-aware matching and input coverage when normalizing source IDs."""
import argparse
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/evaluate_annotations.py'
spec = importlib.util.spec_from_file_location('annotation_evaluator', SCRIPT)
ev = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ev
spec.loader.exec_module(ev)


class AnnotationEvaluatorTests(unittest.TestCase):
    def test_remap_preserves_geometry_and_requires_same_semantic_class(self):
        student = [ev.Box(2, .5, .4, .2, .3)]  # bus in student schema
        raw_gold = [ev.Box(3, .5, .4, .2, .3)]  # bus in supplied gold
        self.assertEqual(ev.match(student, raw_gold, .5), [])
        mapped = ev.remap_boxes(raw_gold, ev.parse_class_map('0:0,1:3,2:1,3:2'))
        self.assertEqual(mapped[0].corners, raw_gold[0].corners)
        self.assertAlmostEqual(ev.match(student, mapped, .5)[0], 1.0)
        self.assertEqual(raw_gold[0].class_id, 3)

    def test_invalid_maps_are_rejected(self):
        for value in ['0:0,1:0,2:1,3:2', '0:0,1:3,2:1', '0:0,1:3,2:1,2:2', 'invalid']:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                ev.parse_class_map(value)

    def test_matching_is_one_to_one_and_keeps_iou_threshold(self):
        box = ev.Box(0, .5, .5, .2, .2)
        self.assertEqual(len(ev.match([box, box], [box], .5)), 1)
        self.assertEqual(ev.match([box], [ev.Box(0, .8, .8, .2, .2)], .5), [])
        self.assertEqual(ev.remap_boxes([box], ev.parse_class_map('0:0,1:1,2:2,3:3')), [box])

    def test_missing_or_extra_label_files_do_not_produce_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold, student = root/'gold', root/'student'
            gold.mkdir(); student.mkdir()
            (gold/'a.txt').write_text('0 .5 .5 .2 .2\n')
            output = root/'result.json'
            command = [sys.executable, str(SCRIPT), '--gold-labels', str(gold),
                       '--student-labels', str(student), '--output', str(output)]
            result = subprocess.run(command, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
            (student/'a.txt').write_text('0 .5 .5 .2 .2\n')
            (student/'extra.txt').write_text('')
            result = subprocess.run(command, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
