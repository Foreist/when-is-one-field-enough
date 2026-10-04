#!/usr/bin/env python3
"""Actual Chromium decoding/resampler checks and fail-closed upload UI tests.

These tests establish equivalence on the bundled PNGs only, not operational
validation on other formats, browsers or physical chips.
"""
import base64
import functools
import http.server
import json
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

    def test_decide_rejects_invalid_probabilities_and_options(self):
        cases=self.page.evaluate('''()=>{
            const probes=[
              ()=>OocPreprocess.decide([NaN]),()=>OocPreprocess.decide([Infinity]),
              ()=>OocPreprocess.decide([null]),()=>OocPreprocess.decide(['0.2']),
              ()=>OocPreprocess.decide([-0.1]),()=>OocPreprocess.decide([1.1]),
              ()=>OocPreprocess.decide([.2],{maxFields:1.5}),
              ()=>OocPreprocess.decide([.2],{maxFields:true}),
              ()=>OocPreprocess.decide([.2],{minFields:NaN}),
              ()=>OocPreprocess.decide([.2],{conf:Infinity}),
              ()=>OocPreprocess.decide([.2],{conf:0}),
              ()=>OocPreprocess.decide([.2],{conf:.5}),
              ()=>OocPreprocess.decide([.2],{conf:1}),
              ()=>OocPreprocess.decide(new DataView(new ArrayBuffer(8)))];
            return probes.map(fn=>{try{fn();return null}catch(e){return e.message}});
        }''')
        self.assertTrue(all(cases),cases)
        audits=self.page.evaluate('''()=>({
          max1:OocPreprocess.decide([.1,.9],{maxFields:1}),
          minAbove:OocPreprocess.decide([.1,.1],{minFields:3}),
          empty:OocPreprocess.decide([]),
          intentional:OocPreprocess.decide([.1,.9]),
          validConf:OocPreprocess.decide([.1],{conf:.500001}),
          typed:OocPreprocess.decide(new Float32Array([.1,.9]))
        })''')
        self.assertEqual(audits['max1']['used'],1)
        self.assertEqual(audits['minAbove']['used'],2)
        self.assertEqual(audits['empty']['total'],0)
        self.assertEqual(audits['intentional']['call'],'inconclusive')

    def test_cached_replay_rejects_malformed_probability_arrays(self):
        server,thread,url=_serve_repo()
        try:
            payloads=[
              '{"p_bad":[]}',
              '{"p_bad":['+','.join(['1e400']*12)+']}',
              '{"p_bad":['+','.join(['null']+['0.1']*11)+']}',
              '{"p_bad":['+','.join(['"0.1"']+['0.1']*11)+']}',
              '{"p_bad":['+','.join(['1.1']+['0.1']*11)+']}',
              '{"p_bad":['+','.join(['0.1']*11+['1e400'])+']}'
            ]
            for payload in payloads:
                page=self.browser.new_page();page.route('https://**/*',lambda route:route.abort())
                page.route('**/demo/examples/good_chip/probs.json',lambda route,request,payload=payload:route.fulfill(status=200,content_type='application/json',body=payload))
                page.goto(url);page.locator('#run').click()
                page.wait_for_function("document.getElementById('status').textContent.includes('failed')")
                state=_ui_state(page)
                self.assertTrue(state['error'],payload)
                self.assertFalse(state['bannerVisible'],payload)
                self.assertFalse(state['tableVisible'],payload)
                self.assertEqual(state['rows'],0,payload)
                self.assertEqual(state['source'],'',payload)
                self.assertTrue(state['replayEnabled'],payload)
                self.assertTrue(state['filesEnabled'],payload)
                page.close()
        finally:
            server.shutdown();server.server_close();thread.join(timeout=3)

    def test_cached_replay_rejects_malformed_manifest_entry(self):
        server,thread,url=_serve_repo()
        base=json.loads((BROWSER/'chips.json').read_text())
        variants=[]
        for files,probs in [('x'*12,'../examples/good_chip/probs.json'),
                            (['ok.png']*12,None),([], '../examples/good_chip/probs.json'),
                            (['']+['ok.png']*11,'../examples/good_chip/probs.json')]:
            manifest=json.loads(json.dumps(base))
            manifest['chips']['good_chip']['files']=files
            manifest['chips']['good_chip']['probs']=probs
            variants.append(json.dumps(manifest))
        try:
            for manifest in variants:
                page=self.browser.new_page();page.route('https://**/*',lambda route:route.abort())
                page.route('**/demo/browser/chips.json',lambda route,request,manifest=manifest:route.fulfill(status=200,content_type='application/json',body=manifest))
                page.goto(url);page.locator('#run').click()
                page.wait_for_function("document.getElementById('status').textContent.includes('failed')")
                state=_ui_state(page)
                self.assertIn('Invalid reference manifest',state['error'])
                self.assertFalse(state['bannerVisible'])
                self.assertFalse(state['tableVisible'])
                self.assertEqual(state['rows'],0)
                self.assertEqual(state['source'],'')
                self.assertTrue(state['replayEnabled'])
                self.assertTrue(state['filesEnabled'])
                page.close()
        finally:
            server.shutdown();server.server_close();thread.join(timeout=3)

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

    def test_controls_have_accessible_labels_and_stable_input_help(self):
        server,thread,url=_serve_repo()
        page=self.browser.new_page()
        page.route('https://**/*',lambda route:route.abort())
        try:
            page.goto(url)
            self.assertEqual(page.get_by_label('Bundled reference example',exact=True).count(),1)
            self.assertEqual(page.get_by_label('PNG fields for exploratory scores',exact=True).count(),1)
            self.assertTrue(page.locator('label[for="chip"]').is_visible())
            self.assertTrue(page.locator('label[for="files"]').is_visible())
            self.assertEqual(page.locator('#files').get_attribute('aria-describedby'),'upload-help')
            help_text=page.locator('#upload-help').inner_text()
            self.assertIn('8-bit grayscale/RGB',help_text)
            self.assertIn('transparency',help_text)
            page.get_by_label('Bundled reference example',exact=True).select_option('bad_chip')
            page.get_by_role('button',name='Replay reference').click()
            page.wait_for_function("document.getElementById('status').textContent.includes('complete')")
            self.assertIn('Reference replay: FAIL',page.locator('#call').inner_text())
            page.evaluate('''()=>{
                window.originalScoreImage=window.scoreImage;
                window.scoreImage=()=>new Promise(resolve=>window.releaseLiveScore=()=>resolve(.123));
            }''')
            page.get_by_label('PNG fields for exploratory scores',exact=True).set_input_files(str(EXAMPLE_PNG))
            page.wait_for_function("typeof window.releaseLiveScore==='function'")
            self.assertEqual(page.locator('#upload-help').inner_text(),help_text)
            self.assertIn('Scoring field',page.locator('#upstatus').inner_text())
            page.evaluate('window.releaseLiveScore()')
            page.wait_for_function("document.getElementById('upstatus').textContent.includes('fields scored')")
            self.assertEqual(page.locator('#upload-help').inner_text(),help_text)
            page.evaluate('delete window.releaseLiveScore;window.scoreImage=window.originalScoreImage;delete window.originalScoreImage')
            page.get_by_label('PNG fields for exploratory scores',exact=True).set_input_files(dict(
                name='unsupported.jpg',mimeType='image/jpeg',buffer=EXAMPLE_PNG.read_bytes()))
            page.wait_for_function("document.getElementById('upstatus').textContent.includes('failed')")
            self.assertEqual(page.locator('#upload-help').inner_text(),help_text)
            self.assertIn('JPEG',page.locator('#error').inner_text())
        finally:
            page.close();server.shutdown();server.server_close();thread.join(timeout=3)

    def test_small_viewports_contain_controls_and_results(self):
        server,thread,url=_serve_repo()
        page=self.browser.new_page()
        page.route('https://**/*',lambda route:route.abort())
        try:
            page.goto(url)
            for width in (320,390):
                with self.subTest(width=width):
                    page.set_viewport_size(dict(width=width,height=844))
                    page.select_option('#chip','borderline_chip')
                    page.get_by_role('button',name='Replay reference').click()
                    page.wait_for_function("document.getElementById('budget').textContent.includes('20 / 20') && !document.getElementById('run').disabled")
                    layout=page.evaluate('''()=>({
                        pageWidth:document.documentElement.scrollWidth,
                        controls:Array.from(document.querySelectorAll('#chip,#files')).map(e=>({
                            left:e.getBoundingClientRect().left,right:e.getBoundingClientRect().right
                        })),
                        rows:document.querySelectorAll('#scores tr').length
                    })''')
                    self.assertEqual(layout['rows'],20)
                    self.assertLessEqual(layout['pageWidth'],width)
                    for control in layout['controls']:
                        self.assertGreaterEqual(control['left'],0)
                        self.assertLessEqual(control['right'],width)
                    wrapper=page.get_by_role('region',name='Field probabilities',exact=True)
                    self.assertEqual(wrapper.locator('table').count(),1)
                    self.assertEqual(wrapper.evaluate('e=>getComputedStyle(e).overflowX'),'auto')
                    self.assertLessEqual(wrapper.bounding_box()['x']+wrapper.bounding_box()['width'],width)
                    wrapper.focus()
                    self.assertTrue(wrapper.evaluate('e=>document.activeElement===e'))
                    page.evaluate("document.querySelector('#scores td').textContent='field_'+'x'.repeat(300)+'.png'")
                    self.assertEqual(page.locator('#scores td').first.evaluate('e=>getComputedStyle(e).overflowWrap'),'anywhere')
                    self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'),width)
                    self.assertIn('Reference replay: PASS',page.locator('#call').inner_text())
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

    def test_invalid_model_outputs_fail_closed_without_results(self):
        server,thread,url=_serve_repo()
        variants=[
          "({logits:{data:[0]}})",
          "({logits:{data:[0,1,2]}})",
          "({})",
          "({logits:{data:[NaN,0]}})",
          "({logits:{data:[Infinity,0],dims:[1,2]}})",
          "({logits:{data:[0,1]}})",
          "({logits:{data:[0,1],dims:[2,1]}})"
        ]
        try:
            for output in variants:
                page=self.browser.new_page();page.route('https://**/*',lambda route:route.abort())
                page.goto(url)
                page.evaluate('''output=>{
                    window.ort={Tensor:function(type,data,shape){this.data=data;this.shape=shape;}};
                    window.model=async()=>({run:async()=>eval(output)});
                }''',output)
                page.locator('#files').set_input_files(str(EXAMPLE_PNG))
                page.wait_for_function("document.getElementById('upstatus').textContent.includes('failed')")
                state=_ui_state(page)
                self.assertIn('Invalid model output',state['error'],output)
                self.assertFalse(state['bannerVisible'],output)
                self.assertFalse(state['tableVisible'],output)
                self.assertEqual(state['rows'],0,output)
                self.assertEqual(state['source'],'',output)
                self.assertTrue(state['replayEnabled'],output)
                self.assertTrue(state['filesEnabled'],output)
                page.close()
        finally:
            server.shutdown();server.server_close();thread.join(timeout=3)

    def test_opaque_png_scores_through_actual_scoreImage_with_stub_model(self):
        server,thread,url=_serve_repo()
        page=self.browser.new_page()
        page.route('https://**/*',lambda route:route.abort())
        try:
            page.goto(url)
            page.evaluate('''()=>{
                window.ort={Tensor:function(type,data,shape){this.data=data;this.shape=shape;}};
                window.model=async()=>({run:async()=>({logits:{data:[Math.log(0.25),Math.log(0.75)],dims:[1,2]}})});
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
