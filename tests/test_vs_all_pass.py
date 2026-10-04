#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('vs_pass',ROOT/'audit/14_vs_all_pass.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class PairedInference(unittest.TestCase):
    def test_repeated_session_has_descriptive_counts_without_iid_p(self):
        c=[dict(session='same',probs=[.9]*8,y=[0]*8),dict(session='same',probs=[.1]*8,y=[1]*8)]
        out=module.compare(c,8)
        self.assertIsNone(out['mcnemar_p']);self.assertEqual(out['n_sessions'],1)
        self.assertEqual(out['inference_status'],'descriptive_only')
        self.assertEqual(out['only_rule_right'],1)

    def test_unknown_session_fail_closed(self):
        out=module.compare([dict(probs=[.9]*8,y=[0]*8)],8)
        self.assertIsNone(out['mcnemar_p'])

    def test_distinct_session_retains_exact_test(self):
        c=[dict(session=str(i),probs=[.9]*8,y=[0]*8) for i in range(5)]
        c.append(dict(session='last',probs=[.9]*8,y=[1]*8))
        out=module.compare(c,8)
        self.assertEqual(out['only_rule_right'],5);self.assertEqual(out['only_all_pass_right'],1)
        self.assertEqual(out['mcnemar_p'],.21875);self.assertEqual(out['n_sessions'],6)

    def test_saved_prediction_array_length_mismatch_fails_closed(self):
        fixture={'p':[.1,.2], 'y':[1], 'session':['s1','s2'], 'idx':[1,2]}
        with self.assertRaisesRegex(ValueError, 'equal lengths'):
            module.chips_from_prediction_arrays(fixture)

    def test_saved_prediction_arrays_reject_invalid_and_duplicate_rows(self):
        base={'p':[.1], 'y':[1], 'session':['s1'], 'idx':[1]}
        cases=[
            dict(base, p=[float('nan')]), dict(base, p=[10**1000]),
            dict(base, p=.5), dict(base, session='s1'), None,
            dict(base, y=[2]),
            dict(base, session=['']), dict(base, idx=[-1]),
            {'p':[.1,.2], 'y':[1,0], 'session':['s1','s1'], 'idx':[1,1]},
        ]
        for fixture in cases:
            with self.subTest(fixture=fixture), self.assertRaises(ValueError):
                module.chips_from_prediction_arrays(fixture)


if __name__=='__main__':unittest.main()
