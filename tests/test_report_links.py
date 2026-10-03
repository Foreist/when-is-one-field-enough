#!/usr/bin/env python3
import re
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from make_report_pdf import Linkify


class ReportLinks(unittest.TestCase):
    def linkify(self, text):
        parser=Linkify();parser.feed(text);return ''.join(parser.parts)

    def test_public_urls_containing_s_and_space_are_not_truncated(self):
        urls=['https://github.com/Foreist/when-is-one-field-enough',
              'https://taewoong23-ooc-chip-qc-demo.static.hf.space/index.html',
              'https://www.youtube.com/watch?v=GMcfX8NgXA0']
        html=self.linkify('<p>'+ ' '.join(urls)+'.</p>')
        self.assertEqual(re.findall(r'href="([^"]+)"',html),urls)

    def test_existing_link_and_code_are_not_relinked(self):
        source='<a href="https://example.com">https://example.com</a><code>https://example.com</code>'
        self.assertEqual(self.linkify(source),source)

    def test_trailing_punctuation_outside_link(self):
        html=self.linkify('<p>(https://example.com/path), done.</p>')
        self.assertIn('<a href="https://example.com/path">https://example.com/path</a>),',html)


if __name__=='__main__':unittest.main()
