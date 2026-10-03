/* Pillow bilinear (antialiased) separable resize for RGB uint8.
 *
 * Numerically matches PIL.Image.resize((w,h), Image.BILINEAR) / torchvision
 * Resize on PIL images (antialias always on for PIL bilinear).
 *
 * Port of Pillow 12.3.0 src/libImaging/Resample.c (horizontal then vertical,
 * Q22 8-bit path, kernel stretched on downsample). Equivalence must be tested
 * through the actual browser decoding path; no validation is inferred here.
 *
 * Pillow / PIL license (MIT-CMU). This file is a reimplementation, not a
 * copy of the C sources:
 *   Copyright (c) 1997-2011 Secret Labs AB
 *   Copyright (c) 1995-2011 Fredrik Lundh and contributors
 *   Copyright (c) 2010 Jeffrey 'Alex' Clark and contributors
 * Permission to use, copy, modify and distribute this software and its
 * documentation for any purpose and without fee is hereby granted, provided
 * that the above copyright notice appears in all copies, and that both that
 * copyright notice and this permission notice appear in supporting
 * documentation, and that the name of Secret Labs AB or the author not be
 * used in advertising or publicity pertaining to distribution of the software
 * without specific, written prior permission.
 * SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
 * SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
 *
 * The Beta posterior uses the exact integer binomial-tail identity at 0.5;
 * it does not incorporate a third-party incomplete-beta implementation.
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
    if (typeof require !== "undefined" && require.main === module) {
      runCli(api);
    }
  } else {
    root.OocPreprocess = api;
  }

  function runCli(api) {
    var fs = require("fs");
    var args = process.argv.slice(2);
    if (args[0] === "--resize-raw") {
      var srcW = +args[1], srcH = +args[2], dstW = +args[3], dstH = +args[4];
      var srcPath = args[5], dstPath = args[6];
      var buf = fs.readFileSync(srcPath);
      var out = api.resizeBilinearPillowUint8(new Uint8Array(buf), srcW, srcH, dstW, dstH);
      fs.writeFileSync(dstPath, Buffer.from(out));
      return;
    }
    process.stderr.write("usage: node preprocess.js --resize-raw srcW srcH dstW dstH in.rgb out.rgb\n");
    process.exit(2);
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  var PRECISION_BITS = 22;
  var SIZE = 384;
  var MEAN = [0.485, 0.456, 0.406];
  var STD = [0.229, 0.224, 0.225];
  var MAXF = 20;
  var MINF = 8;
  var CONF = 0.9;

  // Live uploads must not emit an operational chip call until the host test
  // has signed off JS-vs-PIL on the 44 bundled fields AND ONNX parity.
  // Cached bundled examples are a Python-tool replay, not a live score.
  var LIVE_OPERATIONAL_CHIP_DECISION = false;
  var CACHED_REPLAY_LABEL =
    "cached Python replay on bundled 1024×768 fields — not paper-exact, not full-resolution";
  var LIVE_EXPLORATORY_LABEL =
    "live field scores (exploratory) — not an operational chip decision; no physical chip validated";

  function bilinearFilter(x) {
    if (x < 0.0) x = -x;
    if (x < 1.0) return 1.0 - x;
    return 0.0;
  }

  function precomputeCoeffs(inSize, in0, in1, outSize) {
    var scale = (in1 - in0) / outSize;
    var filterscale = scale < 1.0 ? 1.0 : scale;
    var support = 1.0 * filterscale;
    var ksize = Math.ceil(support) * 2 + 1;
    var kk = new Float64Array(outSize * ksize);
    var bounds = new Int32Array(outSize * 2);
    var inv = 1.0 / filterscale;
    for (var xx = 0; xx < outSize; xx++) {
      var center = in0 + (xx + 0.5) * scale;
      var ww = 0.0;
      var xmin = Math.trunc(center - support + 0.5);
      if (xmin < 0) xmin = 0;
      var xmax = Math.trunc(center + support + 0.5);
      if (xmax > inSize) xmax = inSize;
      xmax -= xmin;
      var kOff = xx * ksize;
      var x = 0;
      for (; x < xmax; x++) {
        var w = bilinearFilter((x + xmin - center + 0.5) * inv);
        kk[kOff + x] = w;
        ww += w;
      }
      if (ww !== 0.0) {
        for (x = 0; x < xmax; x++) kk[kOff + x] /= ww;
      }
      for (; x < ksize; x++) kk[kOff + x] = 0;
      bounds[xx * 2] = xmin;
      bounds[xx * 2 + 1] = xmax;
    }
    return { ksize: ksize, bounds: bounds, kk: kk };
  }

  function normalizeCoeffs8(outCount, ksize, prekk) {
    var scale = 1 << PRECISION_BITS;
    var n = outCount * ksize;
    var kk = new Int32Array(n);
    for (var i = 0; i < n; i++) {
      var w = prekk[i];
      kk[i] = w < 0 ? Math.trunc(-0.5 + w * scale) : Math.trunc(0.5 + w * scale);
    }
    return kk;
  }

  function clip8(ss) {
    var v = ss >> PRECISION_BITS;
    if (v < 0) return 0;
    if (v > 255) return 255;
    return v;
  }

  function packRGB(src, w, h) {
    var packed = new Uint8Array(h * w * 4);
    for (var i = 0, p = 0; i < w * h; i++, p += 3) {
      var o = i * 4;
      packed[o] = src[p];
      packed[o + 1] = src[p + 1];
      packed[o + 2] = src[p + 2];
    }
    return packed;
  }

  function unpackRGB(packed, w, h) {
    var out = new Uint8Array(h * w * 3);
    for (var i = 0, p = 0; i < w * h; i++, p += 3) {
      var o = i * 4;
      out[p] = packed[o];
      out[p + 1] = packed[o + 1];
      out[p + 2] = packed[o + 2];
    }
    return out;
  }

  function resampleHorizontal(src, srcW, srcH, dstW, offset, ksize, bounds, prekk) {
    var kk = normalizeCoeffs8(dstW, ksize, prekk);
    var ROUND = 1 << (PRECISION_BITS - 1);
    var dst = new Uint8Array(srcH * dstW * 4);
    for (var yy = 0; yy < srcH; yy++) {
      var inRow = (yy + offset) * srcW * 4;
      var outRow = yy * dstW * 4;
      for (var xx = 0; xx < dstW; xx++) {
        var xmin = bounds[xx * 2];
        var xmax = bounds[xx * 2 + 1];
        var kOff = xx * ksize;
        var ss0 = ROUND, ss1 = ROUND, ss2 = ROUND;
        for (var x = 0; x < xmax; x++) {
          var kx = kk[kOff + x];
          var pi = inRow + (x + xmin) * 4;
          ss0 += src[pi] * kx;
          ss1 += src[pi + 1] * kx;
          ss2 += src[pi + 2] * kx;
        }
        var po = outRow + xx * 4;
        dst[po] = clip8(ss0);
        dst[po + 1] = clip8(ss1);
        dst[po + 2] = clip8(ss2);
      }
    }
    return dst;
  }

  function resampleVertical(src, srcW, srcH, dstH, ksize, bounds, prekk) {
    var kk = normalizeCoeffs8(dstH, ksize, prekk);
    var ROUND = 1 << (PRECISION_BITS - 1);
    var dst = new Uint8Array(dstH * srcW * 4);
    for (var yy = 0; yy < dstH; yy++) {
      var ymin = bounds[yy * 2];
      var ymax = bounds[yy * 2 + 1];
      var kOff = yy * ksize;
      var outRow = yy * srcW * 4;
      for (var xx = 0; xx < srcW; xx++) {
        var ss0 = ROUND, ss1 = ROUND, ss2 = ROUND;
        for (var y = 0; y < ymax; y++) {
          var ky = kk[kOff + y];
          var pi = (y + ymin) * srcW * 4 + xx * 4;
          ss0 += src[pi] * ky;
          ss1 += src[pi + 1] * ky;
          ss2 += src[pi + 2] * ky;
        }
        var po = outRow + xx * 4;
        dst[po] = clip8(ss0);
        dst[po + 1] = clip8(ss1);
        dst[po + 2] = clip8(ss2);
      }
    }
    return dst;
  }

  function resizeBilinearPillowUint8(srcRGB, srcW, srcH, dstW, dstH) {
    var box = [0.0, 0.0, srcW, srcH];
    var needH = dstW !== srcW || box[0] || box[2] !== dstW;
    var needV = dstH !== srcH || box[1] || box[3] !== dstH;
    var vert = precomputeCoeffs(srcH, box[1], box[3], dstH);
    var yboxFirst = vert.bounds[0];
    var yboxLast = vert.bounds[(dstH - 1) * 2] + vert.bounds[(dstH - 1) * 2 + 1];
    var im = packRGB(srcRGB, srcW, srcH);
    var imW = srcW, imH = srcH;
    if (needH) {
      var horz = precomputeCoeffs(srcW, box[0], box[2], dstW);
      for (var i = 0; i < dstH; i++) vert.bounds[i * 2] -= yboxFirst;
      var sliceH = yboxLast - yboxFirst;
      var slice = new Uint8Array(sliceH * srcW * 4);
      slice.set(im.subarray(yboxFirst * srcW * 4, yboxLast * srcW * 4));
      im = resampleHorizontal(slice, srcW, sliceH, dstW, 0, horz.ksize, horz.bounds, horz.kk);
      imW = dstW;
      imH = sliceH;
    }
    if (needV) {
      im = resampleVertical(im, imW, imH, dstH, vert.ksize, vert.bounds, vert.kk);
      imH = dstH;
    }
    return unpackRGB(im, imW, imH);
  }

  function imageToNchw(rgb, size, mean, std) {
    size = size || SIZE;
    mean = mean || MEAN;
    std = std || STD;
    var t = new Float32Array(3 * size * size);
    var plane = size * size;
    for (var i = 0; i < plane; i++) {
      var p = i * 3;
      t[i] = (rgb[p] / 255 - mean[0]) / std[0];
      t[plane + i] = (rgb[p + 1] / 255 - mean[1]) / std[1];
      t[2 * plane + i] = (rgb[p + 2] / 255 - mean[2]) / std[2];
    }
    return t;
  }

  function rgbToNchw384(srcRGB, srcW, srcH) {
    var resized = resizeBilinearPillowUint8(srcRGB, srcW, srcH, SIZE, SIZE);
    return imageToNchw(resized, SIZE, MEAN, STD);
  }

  function naturalCompare(a, b) {
    var re = /(\d+)/g;
    function parts(s) {
      return String(s).split(re).map(function (t) {
        return /^\d+$/.test(t) ? +t : t.toLowerCase();
      });
    }
    var pa = parts(a), pb = parts(b);
    var n = Math.min(pa.length, pb.length);
    for (var i = 0; i < n; i++) {
      if (pa[i] === pb[i]) continue;
      if (typeof pa[i] === "number" && typeof pb[i] === "number") return pa[i] - pb[i];
      return pa[i] < pb[i] ? -1 : 1;
    }
    return pa.length - pb.length;
  }

  function spreadOrder(n, k) {
    k = k === undefined ? MAXF : k;
    if (k < 1) throw new Error("field budget must be at least one");
    if (n > 0 && k === 1) return [0];
    if (n <= k) {
      var all = [];
      for (var i = 0; i < n; i++) all.push(i);
      return all;
    }
    var out = [];
    for (var j = 0; j < k; j++) out.push(Math.round((j * (n - 1)) / (k - 1)));
    return out;
  }

  // Exact binomial-tail identity for the integer Beta(1+bad,1+good)
  // posterior at 0.5. Budgets here are at most 20, so integers are exact.
  function pBadPosterior(bad, good) {
    if (!Number.isInteger(bad) || !Number.isInteger(good) || bad < 0 || good < 0)
      throw new Error("posterior counts must be nonnegative integers");
    var n = bad + good + 1, choose = 1, sum = 1;
    for (var k = 1; k <= bad; k++) {
      choose = choose * (n - k + 1) / k;
      sum += choose;
    }
    return sum / Math.pow(2, n);
  }

  function decide(probs, opts) {
    opts = opts || {};
    var maxf = opts.maxFields || MAXF;
    var minf = opts.minFields || MINF;
    var conf = opts.conf === undefined ? CONF : opts.conf;
    var order = spreadOrder(probs.length, maxf);
    var bad = 0, good = 0, call = null, used = order.length, p = null;
    for (var i = 0; i < order.length; i++) {
      if (probs[order[i]] > 0.5) bad++; else good++;
      var pb = pBadPosterior(bad, good);
      if (i + 1 >= minf && (pb > conf || 1 - pb > conf)) {
        call = pb > 0.5 ? "fail" : "pass";
        used = i + 1;
        p = pb;
        break;
      }
    }
    if (call === null) {
      p = pBadPosterior(bad, good);
      call = (p > 0.35 && p < 0.65) ? "inconclusive" : (p > 0.5 ? "fail" : "pass");
    }
    return {
      probs: probs,
      call: call,
      p: p,
      used: used,
      total: probs.length,
      order: order.slice(0, used),
      scoredAllFields: true
    };
  }

  function presentResult(source, decision) {
    var cached = source === "cached-replay";
    return {
      source: source,
      operationalChipDecision: false,
      kind: cached ? "replay" : "exploratory",
      label: cached ? CACHED_REPLAY_LABEL : LIVE_EXPLORATORY_LABEL,
      replayCall: cached ? decision.call : null,
      liveCallSuppressed: !cached,
      fieldProbabilities: decision.probs,
      decision: cached ? decision : null
    };
  }

  return {
    SIZE: SIZE, MEAN: MEAN, STD: STD, MAXF: MAXF, MINF: MINF, CONF: CONF,
    PRECISION_BITS: PRECISION_BITS,
    LIVE_OPERATIONAL_CHIP_DECISION: LIVE_OPERATIONAL_CHIP_DECISION,
    CACHED_REPLAY_LABEL: CACHED_REPLAY_LABEL,
    LIVE_EXPLORATORY_LABEL: LIVE_EXPLORATORY_LABEL,
    bilinearFilter: bilinearFilter,
    resizeBilinearPillowUint8: resizeBilinearPillowUint8,
    imageToNchw: imageToNchw,
    rgbToNchw384: rgbToNchw384,
    naturalCompare: naturalCompare,
    spreadOrder: spreadOrder,
    pBadPosterior: pBadPosterior,
    decide: decide,
    presentResult: presentResult
  };
});
