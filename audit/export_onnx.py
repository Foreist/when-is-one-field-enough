#!/usr/bin/env python3
"""Export the shipped checkpoint to ONNX logits and record artifact provenance."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from inference import load_model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,default=ROOT/'model/perfield_mnv3s_384_s0.pt')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    manifest=args.out.with_suffix('.manifest.json')
    if args.out.exists() or manifest.exists():
        raise FileExistsError('choose new ONNX and manifest output paths')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as td:
        staged_out=Path(td)/args.out.name
        staged_manifest=Path(td)/manifest.name
        model=load_model(args.checkpoint)
        tensor=torch.zeros(1,3,384,384)
        torch.onnx.export(model,tensor,str(staged_out),input_names=['input'],output_names=['logits'],
                          opset_version=17,dynamo=False)
        metadata=dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      onnx_sha256=hashlib.sha256(staged_out.read_bytes()).hexdigest(),
                      torch_version=torch.__version__,opset=17,input_shape=[1,3,384,384],
                      outputs='two logits; class 0 is bad',preprocessing='PIL bilinear antialias RGB resize; ImageNet normalization')
        staged_manifest.write_text(json.dumps(metadata,indent=2)+'\n')
        os.link(staged_out,args.out)
        try:
            os.link(staged_manifest,manifest)
        except BaseException:
            staged_stat=staged_out.stat()
            try:
                published_stat=args.out.stat()
            except FileNotFoundError:
                pass
            else:
                if (published_stat.st_dev,published_stat.st_ino)==(staged_stat.st_dev,staged_stat.st_ino):
                    args.out.unlink()
            raise
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
