#!/usr/bin/env python3
"""Mutations exercise row/column completeness as well as surviving cell values."""
import re
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from check_tables import check_tables


class TableCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.text=(ROOT/'REPORT.md').read_text()

    def test_current_tables(self):self.assertEqual(check_tables(self.text),[])

    def test_in_distribution_baseline_cells_checked(self):
        changed,n=re.subn(r'(\| in-distribution[^\n]*\| )0\.734( \| 0\.733 \| )0\.791',r'\g<1>0.999\g<2>0.123',self.text,count=1)
        self.assertEqual(n,1)
        errors=check_tables(changed)
        self.assertTrue(any('baseline acc' in e for e in errors),errors)
        self.assertTrue(any('baseline auc' in e for e in errors),errors)

    def test_missing_age_row(self):
        changed,n=re.subn(r'^\| 0-1_days[^\n]*\n','',self.text,count=1,flags=re.M)
        self.assertEqual(n,1);self.assertTrue(any('wrong row IDs' in e for e in check_tables(changed)))

    def test_duplicate_age_row(self):
        line=next(x for x in self.text.splitlines() if x.startswith('| 0-1_days'))
        self.assertTrue(any('duplicates' in e for e in check_tables(self.text.replace(line,line+'\n'+line,1))))

    def test_inner_cv_all_fields_column(self):
        changed,n=re.subn(r'(\| fields per chip \| 5\.3 \| 8\.1 \| 9\.9 \| )24\.9',r'\g<1>99.9',self.text,count=1)
        self.assertEqual(n,1);self.assertTrue(any('InnerCV all fields' in e for e in check_tables(changed)))

    def test_policy_table_row_and_subcell_count(self):
        changed=self.text.replace('| 4 | 0.791 / 0.335 |','| 4 | 0.791 |',1)
        self.assertNotEqual(changed,self.text);self.assertTrue(any('subcells' in e for e in check_tables(changed)))

    def test_recovery_unknown_label(self):
        changed=self.text.replace('| run length only |','| unknown feature |',1)
        self.assertTrue(any('Recovery: wrong row IDs' in e for e in check_tables(changed)))


if __name__=='__main__':unittest.main()
