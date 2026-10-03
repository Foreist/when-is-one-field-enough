#!/usr/bin/env python3
"""Previously green cross-document and whole-token mutations must fail."""
import copy
import json
import re
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import check_claims


class ClaimCoverage(unittest.TestCase):
    def setUp(self):
        self.ledger=check_claims.load_ledger(ROOT/'claims.json')
        self.texts=check_claims.read_texts(self.ledger,ROOT)

    def test_malformed_comma_grouping_cannot_equal_valid_literal(self):
        self.assertFalse(check_claims.lits_match('12,34','1234'))
        self.assertFalse(check_claims.lits_match('1,,234','1,234'))
        self.assertTrue(check_claims.lits_match('1,234','1234'))
        self.assertTrue(check_claims.lits_match('+7.9','7.9'))

    def test_whole_number_suffix_and_exponent(self):
        pattern=r'(chip accuracy \*\*among called chips\*\*[^\n]*?\*\*)0\.826'
        for wrong in ['0.8269','0.826e2','0.8261','0.8260']:
            texts=dict(self.texts);texts['readme'],n=re.subn(pattern,lambda m:m.group(1)+wrong,texts['readme'],count=1)
            self.assertEqual(n,1)
            self.assertTrue(check_claims.check_claims(ROOT,self.ledger,texts),wrong)

    def test_correct_writeup_cannot_hide_report_missing_anchor(self):
        if self.texts.get('writeup') is None:self.skipTest('workspace writeup not available')
        texts=dict(self.texts)
        texts['report'],n=re.subn(r'(every\*\* field \(0\.800\)[\s\S]{0,80}?\*\*)9\.5',r'\g<1>9.59',texts['report'],count=1)
        self.assertEqual(n,1)
        errors=check_claims.check_claims(ROOT,self.ledger,texts)
        self.assertTrue(any('instead_of' in e and 'report' in e for e in errors),errors)

    def test_missing_doc_count_schema_fails(self):
        ledger=copy.deepcopy(self.ledger)
        ledger['claims'][0]['anchors'][0].pop('counts')
        self.assertTrue(any('no expected count' in e for e in check_claims.check_claims(ROOT,ledger,self.texts)))

    def test_extra_same_claim_occurrence_fails_count(self):
        texts=dict(self.texts);texts['report']+='\n19/23 = 0.826\n'
        self.assertTrue(any('report_19_23' in e and 'need exact' in e for e in check_claims.check_claims(ROOT,self.ledger,texts)))


if __name__=='__main__':unittest.main()
