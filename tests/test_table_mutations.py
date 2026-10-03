#!/usr/bin/env python3
"""Enumerate every numeric token and row in the covered result tables."""
import re
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from check_tables import check_tables

PREFIXES=['| session | fields | good share','| culture age','| cell line | fields | single',
          '| budget k | random: acc / recall / err','| budget k | random: acc / recall |',
          '| feature set | CV AUC','| budget k | mean: acc / sens / spec','| on 68 non-test chips',
          '| metric | value |','| policy | fields per chip','| chip | fields | bad share | field acc |','| held-out cell line']
NUMBER=re.compile(r'(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?(?![\w])',re.I)


def spans(text):
    in_table=False; header_lines=0;pos=0
    for line in text.splitlines(keepends=True):
        if any(line.startswith(p) for p in PREFIXES):in_table=True;header_lines=2
        if in_table and not line.startswith('|'):in_table=False
        if in_table:
            if header_lines:header_lines-=1
            else:yield pos,line
        pos+=len(line)


class ExhaustiveTableCanaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.text=(ROOT/'REPORT.md').read_text()

    def test_each_numeric_token_is_checked(self):
        misses=[];count=0
        for offset,line in spans(self.text):
            for match in NUMBER.finditer(line):
                old=match.group();new=old[:-1]+('1' if old[-1]=='0' else '0')
                changed=self.text[:offset+match.start()]+new+self.text[offset+match.end():]
                count+=1
                try:errors=check_tables(changed)
                except (ValueError,KeyError,IndexError,TypeError):continue
                if not errors:misses.append((line.strip(),old,new))
        self.assertGreater(count,250)
        self.assertEqual(misses,[],f'{len(misses)}/{count} unchecked numeric cells: {misses[:12]}')

    def test_deleting_any_required_row_fails(self):
        misses=[];count=0
        for offset,line in spans(self.text):
            # Table7 packs two entries; deleting a packed row must also fail.
            changed=self.text[:offset]+self.text[offset+len(line):];count+=1
            try:errors=check_tables(changed)
            except (ValueError,KeyError,IndexError,TypeError):continue
            if not errors:misses.append(line.strip())
        self.assertGreater(count,50);self.assertEqual(misses,[],misses)


if __name__=='__main__':unittest.main()
