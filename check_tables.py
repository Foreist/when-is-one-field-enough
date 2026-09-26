#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cell-by-cell check of the REPORT.md tables against results/*.json.

check_numbers.py only asks whether a number exists *somewhere* in the results; a table cell can
pass that by coincidence. Here every cell of Tables 1-5, 7, 8 and the inner-CV table is compared
with the exact JSON leaf it was written from. Exit code 1 on any mismatch.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = lambda f: json.loads((HERE / "results" / f).read_text())
t = (HERE / "REPORT.md").read_text().split("\n")
def rows(start_pat, n=None):
    i=next(j for j,l in enumerate(t) if l.startswith(start_pat))
    out=[]
    for l in t[i:]:
        if not l.startswith('|'): break
        out.append([c.strip().strip('*') for c in l.strip('|').split('|')])
    return out[2:]
bad=0
def eq(label, text, val, fmt):
    global bad
    s=format(val,fmt)
    if text!=s: bad+=1; print('MISMATCH',label,text,'vs',s)
# Table 1
sp={r['session']:r for r in R('structure_probe.json')['A']}
for r in rows('| session | fields | good share'):
    a=sp[r[0]]; eq('T1 n '+r[0],r[1],a['n'],'d'); eq('T1 gf '+r[0],r[2],a['good_frac'],'.2f'); eq('T1 runs',r[3],a['runs'],'d'); eq('T1 exp '+r[0],r[4],a['expected'],'.1f'); eq('T1 z '+r[0],r[5],a['z'],'.2f')
# Table 2
ls=R('label_sufficiency.json')
for r in rows('| culture age'):
    d=ls['single_image_agreement_by_day'][r[0]]; eq('T2 day n',r[1],d['n'],'d'); eq('T2 day '+r[0],r[2],d['agree'],'.3f')
for r in rows('| cell line | fields | single'):
    d=ls['single_image_agreement_by_cell']['cell_type_'+r[0]]; eq('T2 cell n',r[1],d['n'],'d'); eq('T2 '+r[0],r[2],d['agree'],'.3f')
# Table 3
a=R('adaptive_sampling.json')
for r in rows('| budget k | random: acc / recall / err'):
    for p,c in zip(['random','window','adaptive'],r[1:]):
        x=[s.strip() for s in c.split('/')]; d=a[r[0]][p]
        for s,key in zip(x,['chip_acc','bad_recall','frac_err']): eq(f'T3 {r[0]} {p} {key}',s,d[key],'.3f')
# Table 4
m=R('policy_model_in_loop.json')
for r in rows('| budget k | random: acc / recall |'):
    for p,c in zip(['random','window','adaptive'],r[1:]):
        x=[s.strip() for s in c.split('/')]; d=m[r[0]][p]
        eq(f'T4 {r[0]} {p} acc',x[0],d['chip_acc'],'.3f'); eq(f'T4 {r[0]} {p} rec',x[1],d['bad_recall'],'.3f')
# recovery table
rc=R('recovery_test.json')
for r in rows('| feature set | CV AUC'):
    k={'run length only':'length only','artifact signals only':'artifact signals only','run length + artifacts':'length + artifact','all features':'all'}.get(r[0])
    if k: eq('rec '+r[0],r[1],rc['cv_auc'][k],'.3f')
# Table 5
g=R('aggregation_results.json')['summary']
for r in rows('| budget k | mean: acc / sens / spec'):
    for p,c in zip(['mean','max','top2'],r[1:]):
        x=[s.strip() for s in c.split('/')]; d=g[r[0]][p]
        for s,key in zip(x,['acc','sens','spec']): eq(f'T5 {r[0]} {p} {key}',s,d[key],'.3f')
# Table 8
lo=R('lolo_cellline.json')['per_cell']
for r in rows('| held-out cell line'):
    if r[0].startswith('mean') or r[0].startswith('in-dist'): continue
    d=lo['cell_type_'+r[0]]; n,s=r[1].replace(')','').split(' (')
    eq('T8 n '+r[0],n,d['n_test'],'d'); eq('T8 s '+r[0],s,d['n_sessions_test'],'d')
    eq('T8 acc '+r[0],r[2],d['acc'],'.3f'); eq('T8 bal '+r[0],r[3],d['bal_acc'],'.3f'); eq('T8 auc '+r[0],r[4],d['auc'],'.3f')
mr=[r for r in rows('| held-out cell line') if r[0].startswith('mean')][0]
for i,k in [(2,'acc'),(3,'bal_acc'),(4,'auc')]: eq('T8 mean '+k,mr[i],np.mean([v[k] for v in lo.values()]),'.3f')
# Table 7
pc=R('per_chip_calls.json')['per_session']
for r in rows('| chip (session) | fields'):
    d=pc[r[0]]; eq('T7 n '+r[0],r[1],d['n_fields'],'d'); eq('T7 bs '+r[0],r[2],f"{100*d['bad_share']:.0f}%",'s'); eq('T7 acc '+r[0],r[3],d['field_acc'],'.3f')
# inner CV table
ic=R('inner_cv_minfields.json')
rr=rows('| on 68 non-test chips')
for i,k in [(1,'1'),(2,'8'),(3,'12')]:
    d=ic['by_min_fields'][k]; eq('cv acc '+k,rr[0][i],d['force_acc'],'.3f'); eq('cv fc '+k,rr[1][i],d['false_confident'],'d'); eq('cv f '+k,rr[2][i],d['mean_fields'],'.1f')
print(f'{bad} table mismatches', file=sys.stderr)
sys.exit(1 if bad else 0)
