#!/usr/bin/env python3
"""Actual Chromium decoding/resampler checks and fail-closed upload UI tests.

These tests establish equivalence on the bundled PNGs only, not operational
validation on other formats, browsers or physical chips.
"""
import base64
import functools
import http.server
import tempfile
import threading
import unittest
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
BROWSER=ROOT/'demo/browser'
EXAMPLE_PNG=next((ROOT/'demo/examples/good_chip').glob('*.png'))


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args):
        pass


def _serve_repo():
    server=http.server.ThreadingHTTPServer(
        ('127.0.0.1',0),functools.partial(QuietHandler,directory=str(ROOT)))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    url=f'http://127.0.0.1:{server.server_address[1]}/demo/browser/index.html'
    return server,thread,url


def _write_temp_image(suffix,image,fmt,**save_kwargs):
    handle=tempfile.NamedTemporaryFile(suffix=suffix,delete=False)
    image.save(handle.name,format=fmt,**save_kwargs)
    handle.close()
    return handle.name


def _ui_state(page):
    return page.evaluate('''()=>({
        bannerVisible:!document.getElementById('result').hidden,
        banner:document.getElementById('call').textContent,
        tableVisible:!document.getElementById('field-results').hidden,
        rows:document.querySelectorAll('#scores tr').length,
        source:document.getElementById('source-label').textContent,
        error:document.getElementById('error').textContent,
        uploadStatus:document.getElementById('upstatus').textContent,
        replayEnabled:!document.getElementById('run').disabled,
        filesEnabled:!document.getElementById('files').disabled
    })''')


class BrowserPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pw=sync_playwright().start()
        cls.browser=cls.pw.chromium.launch(args=['--no-sandbox'])
        cls.page=cls.browser.new_page()
        cls.page.set_content('<!doctype html><html><body>resampler check</body></html>')
        cls.page.add_script_tag(path=str(BROWSER/'preprocess.js'))

    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.pw.stop()

    def test_raw_image_path_matches_pillow_for_44_bundled_pngs(self):
        files=sorted((ROOT/'demo/examples').glob('*/*.png'))
        self.assertEqual(len(files),44)
        for file in files:
            expected=Image.open(file).convert('RGB').resize((384,384),Image.Resampling.BILINEAR).tobytes()
            args=dict(png=base64.b64encode(file.read_bytes()).decode(),expected=base64.b64encode(expected).decode())
            diff=self.page.evaluate('''async ({png,expected})=>{
                const src=Uint8Array.from(atob(png),c=>c.charCodeAt(0));
                const im=await createImageBitmap(new Blob([src],{type:'image/png'}),{colorSpaceConversion:'none'});
                const canvas=document.createElement('canvas');canvas.width=im.width;canvas.height=im.height;
                const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(im,0,0);
                const rgba=ctx.getImageData(0,0,im.width,im.height).data;
                const rgb=new Uint8Array(im.width*im.height*3);
                for(let i=0;i<im.width*im.height;i++)for(let ch=0;ch<3;ch++)rgb[i*3+ch]=rgba[i*4+ch];
                const actual=OocPreprocess.resizeBilinearPillowUint8(rgb,im.width,im.height,384,384);
                const want=Uint8Array.from(atob(expected),c=>c.charCodeAt(0));let max=0,count=0;
                for(let i=0;i<want.length;i++){const d=Math.abs(want[i]-actual[i]);if(d){count++;max=Math.max(max,d);}}
                im.close();return {max,count};
            }''',args)
            self.assertEqual(diff,dict(max=0,count=0),str(file.relative_to(ROOT))+str(diff))

    def test_cached_and_exploratory_presentation_are_separate(self):
        result=self.page.evaluate('''()=>{
            const d=OocPreprocess.decide(Array(12).fill(.1));
            return {cached:OocPreprocess.presentResult('cached-replay',d),live:OocPreprocess.presentResult('live-upload',d)};
        }''')
        self.assertEqual(result['cached']['replayCall'],'pass')
        self.assertIsNone(result['live']['replayCall'])
        self.assertIsNone(result['live']['decision'])
        self.assertTrue(result['live']['liveCallSuppressed'])
        self.assertFalse(result['live']['operationalChipDecision'])

    def test_exact_beta_tail_and_one_field_budget(self):
        result=self.page.evaluate('''()=>({
            good3:OocPreprocess.pBadPosterior(0,3),bad3:OocPreprocess.pBadPosterior(3,0),
            tie:OocPreprocess.pBadPosterior(10,10),one:OocPreprocess.spreadOrder(100,1)
        })''')
        self.assertEqual(result,dict(good3=0.0625,bad3=0.9375,tie=0.5,one=[0]))

    def test_natural_order_and_spread_prefix(self):
        result=self.page.evaluate('''()=>({
            names:['x_10.png','x_2.png','x_1.png'].sort(OocPreprocess.naturalCompare),
            consumed:OocPreprocess.decide(Array(100).fill(.1)).order
        })''')
        self.assertEqual(result['names'],['x_1.png','x_2.png','x_10.png'])
        self.assertEqual(result['consumed'],[0,5,10,16,21,26,31,36])

    def test_default_cached_button_replays_without_loading_model(self):
        server,thread,url=_serve_repo()
        page=self.browser.new_page()
        page.route('https://**/*',lambda route:route.abort())
        try:
            page.goto(url)
            page.get_by_role('button',name='Replay reference').click()
            page.wait_for_function("document.getElementById('status').textContent.includes('complete')")
            self.assertIn('Reference replay: PASS',page.locator('#call').inner_text())
            self.assertIn('8 / 12',page.locator('#budget').inner_text())
            self.assertIn('no model inference',page.locator('#status').inner_text())
            self.assertEqual(page.locator('#error').inner_text(),'')
            page.select_option('#chip','borderline_chip')
            page.get_by_role('button',name='Replay reference').click()
            page.wait_for_function("document.getElementById('budget').textContent.includes('20 / 20')")
            self.assertIn('PASS',page.locator('#call').inner_text())
            self.assertIn('0.668',page.locator('#call').inner_text())
        finally:
            page.close();server.shutdown();server.server_close();thread.join(timeout=3)

    def test_network_blocked_upload_after_replay_clears_cached_rows(self):
        server,thread,url=_serve_repo()
        page=self.browser.new_page()
        page.route('https://**/*',lambda route:route.abort())
        try:
            page.goto(url)
            page.get_by_role('button',name='Replay reference').click()
            page.wait_for_function("document.getElementById('status').textContent.includes('complete')")
            page.locator('#files').set_input_files(str(EXAMPLE_PNG))
            page.wait_for_function("document.getElementById('upstatus').textContent.includes('failed')")
            state=_ui_state(page)
            self.assertFalse(state['bannerVisible'])
            self.assertFalse(state['tableVisible'])
            self.assertEqual(state['rows'],0)
            self.assertEqual(state['source'],'')
            self.assertTrue(state['error'])
            self.assertTrue(state['replayEnabled'])
        finally:
            page.close();server.shutdown();server.server_close();thread.join(timeout=3)

    def test_inflight_live_disables_replay_and_finish_has_no_stale_banner(self):
        server,thread,url=_serve_repo()
        page=self.browser.new_page()
        page.route('https://**/*',lambda route:route.abort())
        try:
            page.goto(url)
            page.evaluate('''()=>{
                window.scoreImage=()=>new Promise(resolve=>window.releaseLiveScore=()=>resolve(.123));
            }''')
            page.locator('#files').set_input_files(str(EXAMPLE_PNG))
            page.wait_for_function("typeof window.releaseLiveScore==='function'")
            self.assertFalse(page.locator('#run').is_enabled())
            self.assertFalse(page.locator('#files').is_enabled())
            page.get_by_role('button',name='Replay reference').click(force=True)
            page.evaluate('window.replay()')
            inflight=_ui_state(page)
            self.assertFalse(inflight['bannerVisible'])
            self.assertEqual(inflight['rows'],0)
            page.evaluate('window.releaseLiveScore()')
            page.wait_for_function("document.getElementById('upstatus').textContent.includes('fields scored')")
            done=_ui_state(page)
            self.assertFalse(done['bannerVisible'])
            self.assertEqual(done['banner'],'')
            self.assertNotIn('PASS',done['banner'])
            self.assertTrue(done['tableVisible'])
            self.assertEqual(done['rows'],1)
            self.assertIn('exploratory',done['source'].lower())
            self.assertIn('0.12300',page.locator('#scores').inner_text())
            self.assertTrue(done['replayEnabled'])
        finally:
            page.close();server.shutdown();server.server_close();thread.join(timeout=3)

    def test_transparent_png_is_rejected_without_cached_rows(self):
        server,thread,url=_serve_repo()
        page=self.browser.new_page()
        page.route('https://**/*',lambda route:route.abort())
        path=_write_temp_image('.png',Image.new('RGBA',(32,16),(255,80,20,0)),'PNG')
        try:
            page.goto(url)
            page.get_by_role('button',name='Replay reference').click()
            page.wait_for_function("document.getElementById('status').textContent.includes('complete')")
            page.locator('#files').set_input_files(path)
            page.wait_for_function("document.getElementById('upstatus').textContent.includes('failed')")
            state=_ui_state(page)
            self.assertIn('Unsupported input',state['error'])
            self.assertIn('transparen',state['error'].lower())
            self.assertEqual(state['rows'],0)
            self.assertFalse(state['bannerVisible'])
            self.assertFalse(state['tableVisible'])
        finally:
            Path(path).unlink(missing_ok=True)
            page.close();server.shutdown();server.server_close();thread.join(timeout=3)

    def test_jpeg_is_rejected_without_decoding_as_live_input(self):
        server,thread,url=_serve_repo()
        page=self.browser.new_page()
        page.route('https://**/*',lambda route:route.abort())
        jpeg=Image.new('RGB',(48,24))
        for y in range(24):
            for x in range(48):
                jpeg.putpixel((x,y),(x*5,y*10,(x+y)*3))
        path=_write_temp_image('.jpg',jpeg,'JPEG')
        try:
            page.goto(url)
            page.locator('#files').set_input_files(path)
            page.wait_for_function("document.getElementById('upstatus').textContent.includes('failed')")
            state=_ui_state(page)
            self.assertIn('Unsupported input',state['error'])
            self.assertIn('JPEG',state['error'])
            self.assertEqual(state['rows'],0)
            self.assertFalse(state['tableVisible'])
        finally:
            Path(path).unlink(missing_ok=True)
            page.close();server.shutdown();server.server_close();thread.join(timeout=3)

    def test_unsupported_png_encodings_rejected_by_actual_path(self):
        server,thread,url=_serve_repo()
        page=self.browser.new_page();page.route('https://**/*',lambda route:route.abort())
        try:
            page.goto(url)
            raw=EXAMPLE_PNG.read_bytes()
            ihdr=bytearray(raw);ihdr[24]=16
            import struct,zlib
            chunk_type=b'acTL';data=struct.pack('>II',2,0)
            chunk=struct.pack('>I',len(data))+chunk_type+data+struct.pack('>I',zlib.crc32(chunk_type+data)&0xffffffff)
            animated=raw[:33]+chunk+raw[33:]
            prof_type=b'sRGB';prof_data=b'\0'
            profile=struct.pack('>I',len(prof_data))+prof_type+prof_data+struct.pack('>I',zlib.crc32(prof_type+prof_data)&0xffffffff)
            profiled=raw[:33]+profile+raw[33:]
            trns_type=b'tRNS';trns_data=struct.pack('>H',65535)
            trns_chunk=struct.pack('>I',len(trns_data))+trns_type+trns_data+struct.pack('>I',zlib.crc32(trns_type+trns_data)&0xffffffff)
            transparent_key=raw[:33]+trns_chunk+raw[33:]
            for label,data in [('depth16',bytes(ihdr)),('animation',animated),('profile',profiled),('transparent-key',transparent_key)]:
                encoded=base64.b64encode(data).decode()
                outcome=page.evaluate('''async raw=>{
                    const bytes=Uint8Array.from(atob(raw),c=>c.charCodeAt(0));
                    try{await scoreImage(new Blob([bytes],{type:'image/png'}));return 'accepted';}
                    catch(e){return e.message;}
                }''',encoded)
                self.assertIn('Unsupported input',outcome,label+':'+outcome)
        finally:
            page.close();server.shutdown();server.server_close();thread.join(timeout=3)

    def test_opaque_png_scores_through_actual_scoreImage_with_stub_model(self):
        server,thread,url=_serve_repo()
        page=self.browser.new_page()
        page.route('https://**/*',lambda route:route.abort())
        try:
            page.goto(url)
            page.evaluate('''()=>{
                window.ort={Tensor:function(type,data,shape){this.data=data;this.shape=shape;}};
                window.model=async()=>({run:async()=>({logits:{data:[Math.log(0.25),Math.log(0.75)]}})});
            }''')
            page.locator('#files').set_input_files(str(EXAMPLE_PNG))
            page.wait_for_function("document.getElementById('upstatus').textContent.includes('fields scored')")
            state=_ui_state(page)
            self.assertEqual(state['error'],'')
            self.assertFalse(state['bannerVisible'])
            self.assertEqual(state['banner'],'')
            self.assertTrue(state['tableVisible'])
            self.assertEqual(state['rows'],1)
            self.assertIn('exploratory',state['source'].lower())
            self.assertIn('0.25000',page.locator('#scores').inner_text())
            self.assertIn('No operational chip call',state['uploadStatus'])
        finally:
            page.close();server.shutdown();server.server_close();thread.join(timeout=3)


if __name__=='__main__':unittest.main()
