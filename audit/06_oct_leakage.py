#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is the OoC QC leakage unique? Audit a second organoid imaging benchmark.

Dataset: "Segmentation and Multi-Timepoint Tracking of 3D Cancer Organoids from Optical
Coherence Tomography" (zenodo.15783866, CC-BY-4.0, Diagnostics 2024).
File names encode the acquisition group:
    train/val :  w<well>_d<day>_<slice>.png
    test      :  d<day>_p<plate>_w<well>_<slice>.png
The same (well, day) means the same organoids imaged in the same session.

We read the archive's central directory over HTTP range requests (no 4.9 GB download) and
report how many test fields share a (well, day) group with training.

Writes results/oct_leakage.json
"""
import collections, json, re, struct, urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results"
RECORD = "15783866"
PAT_TRAIN = re.compile(r"^w(\d+)_d(\d+)_(\d+)\.png$")
PAT_TEST = re.compile(r"^d(\d+)_p(\d+)_w(\d+)_(\d+)\.png$")


def zip_names(url, size, tail_bytes=2_000_000):
    def rng(a, b):
        req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
        return urllib.request.urlopen(req, timeout=120).read()
    tail = rng(max(0, size - tail_bytes), size - 1)
    j = tail.rfind(b"PK\x06\x06")
    if j >= 0:
        z = tail[j:j + 56]
        cd_size = struct.unpack("<Q", z[40:48])[0]
        cd_off = struct.unpack("<Q", z[48:56])[0]
    else:
        i = tail.rfind(b"PK\x05\x06")
        e = tail[i:i + 22]
        cd_size = struct.unpack("<I", e[12:16])[0]
        cd_off = struct.unpack("<I", e[16:20])[0]
    cd = rng(cd_off, cd_off + cd_size - 1)
    names, p = [], 0
    while p < len(cd) - 4:
        if cd[p:p + 4] != b"PK\x01\x02":
            break
        nl = struct.unpack("<H", cd[p + 28:p + 30])[0]
        el = struct.unpack("<H", cd[p + 30:p + 32])[0]
        cl = struct.unpack("<H", cd[p + 32:p + 34])[0]
        names.append(cd[p + 46:p + 46 + nl].decode("utf-8", "replace"))
        p += 46 + nl + el + cl
    return names


def main():
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{RECORD}", timeout=60) as r:
        rec = json.load(r)
    f = [x for x in rec["files"] if x["key"].endswith(".zip")][0]
    print(f"{rec['metadata']['title'][:80]}  ({f['size']/1e9:.2f} GB, licence "
          f"{(rec['metadata'].get('license') or {}).get('id')})")
    names = zip_names(f["links"]["self"], f["size"])
    print("entries:", len(names))

    groups = collections.defaultdict(set)
    n_files = collections.Counter()
    leak_files = 0
    for x in names:
        parts = x.split("/")
        if len(parts) < 4:
            continue
        split, fn = parts[1], parts[-1]
        m = PAT_TRAIN.match(fn)
        if m:
            groups[split].add((m.group(1), m.group(2)))
            n_files[split] += 1
            continue
        m = PAT_TEST.match(fn)
        if m:
            groups[split].add((m.group(3), m.group(1)))
            n_files[split] += 1
    for s in groups:
        print(f"  {s}: {n_files[s]} files, {len(groups[s])} (well,day) groups")

    tr, va, te = groups["train"], groups["val"], groups["test"]
    # recount test files that fall in a training group
    for x in names:
        parts = x.split("/")
        if len(parts) < 4 or parts[1] != "test":
            continue
        m = PAT_TEST.match(parts[-1])
        if m and (m.group(3), m.group(1)) in tr:
            leak_files += 1

    out = dict(source=f"zenodo.{RECORD}", title=rec["metadata"]["title"],
               licence=(rec["metadata"].get("license") or {}).get("id"),
               files_per_split=dict(n_files), groups_per_split={k: len(v) for k, v in groups.items()},
               train_groups=sorted(tr), val_groups=sorted(va), test_groups=sorted(te),
               train_cap_test=sorted(tr & te), train_cap_val=sorted(tr & va), val_cap_test=sorted(va & te),
               test_files=int(n_files["test"]),
               test_files_in_train_group=int(leak_files),
               frac_test_files_in_train_group=leak_files / n_files["test"],
               note="the same (well, day) = the same organoids imaged in the same session; for a "
                    "tracking benchmark the same instances therefore appear on both sides")
    (OUT / "oct_leakage.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\ntrain ∩ test groups: {len(tr & te)}  {sorted(tr & te)}")
    print(f"val ∩ test groups:   {len(va & te)}  {sorted(va & te)}")
    print(f"test files in a (well,day) also present in training: {leak_files}/{n_files['test']} "
          f"({100*leak_files/n_files['test']:.1f}%)")
    print("saved", OUT / "oct_leakage.json")


if __name__ == "__main__":
    main()
