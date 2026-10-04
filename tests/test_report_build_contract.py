#!/usr/bin/env python3
"""Failed report rendering must not overwrite verified PDF/manifest artifacts."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import make_report_pdf as renderer


class ReportBuildContract(unittest.TestCase):
    def run_fixture(self,source_changed=False,render_failure=False):
        td=tempfile.TemporaryDirectory()
        root=Path(td.name)
        (root/'REPORT.md').write_text('# Fixture\n\nStable source')
        pdf=root/'report.pdf';pdf.write_bytes(b'%PDF-old-verified')
        sidecar=root/'report.sources.json';sidecar.write_bytes(b'{"old":true}')
        old_pdf=pdf.read_bytes();old_manifest=sidecar.read_bytes()
        closed=[]

        class Page:
            def goto(self,url):pass
            def pdf(self,path,**kwargs):
                Path(path).write_bytes(b'%PDF-new-render')
                if render_failure:raise RuntimeError('browser rendering failed')
        class Browser:
            def new_page(self):return Page()
            def close(self):closed.append(True)
        class Playwright:
            def __init__(self):self.chromium=self
            def launch(self):return Browser()
            def __enter__(self):return self
            def __exit__(self,*args):pass

        manifests=[{'REPORT.md':'before'},{'REPORT.md':'after' if source_changed else 'before'}]
        try:
            with mock.patch.object(renderer,'HERE',root),mock.patch.object(renderer,'MD',root/'REPORT.md'),mock.patch.object(renderer,'PDF',pdf),mock.patch.object(renderer,'sync_playwright',lambda:Playwright()),mock.patch.object(renderer,'source_manifest',side_effect=manifests):
                if source_changed or render_failure:
                    with self.assertRaises(RuntimeError):renderer.main()
                    self.assertEqual(pdf.read_bytes(),old_pdf)
                    self.assertEqual(sidecar.read_bytes(),old_manifest)
                else:
                    renderer.main()
                    self.assertEqual(pdf.read_bytes(),b'%PDF-new-render')
                    manifest=json.loads(sidecar.read_text())
                    self.assertEqual(manifest['sources'],{'REPORT.md':'before'})
                    self.assertEqual(manifest['pdf_sha256'],renderer.hashlib.sha256(pdf.read_bytes()).hexdigest())
            self.assertTrue(closed)
            self.assertEqual(sorted(p.name for p in root.iterdir()),['REPORT.md','report.pdf','report.sources.json'])
        finally:td.cleanup()

    def test_source_drift_preserves_existing_verified_pair(self):
        self.run_fixture(source_changed=True)

    def test_render_exception_preserves_existing_verified_pair_and_cleans_temp(self):
        self.run_fixture(render_failure=True)

    def test_success_publishes_matching_pdf_and_manifest(self):
        self.run_fixture()


if __name__=='__main__':unittest.main()
