#!/usr/bin/env python3
"""Validate numeric report tables, including their expected rows and columns.

The checked tables are explicitly enumerated below. Missing/extra/duplicate rows,
unknown labels and altered baseline/aggregate values fail; prose is separate.
"""
import argparse
import collections
import json
import math
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent


def check_tables(text=None, here=HERE):
    here=Path(here)
    lines=(text if text is not None else (here/'REPORT.md').read_text()).splitlines()
    issues=[]
    def load(name):return json.loads((here/'results'/name).read_text())
    def table(prefix,width):
        starts=[i for i,line in enumerate(lines) if line.startswith(prefix)]
        if len(starts)!=1:
            issues.append(f'TABLE {prefix}: {len(starts)} headers, need exactly one');return []
        rows=[]
        for line in lines[starts[0]+2:]:
            if not line.startswith('|'):break
            row=[]
            for cell in line.strip().strip('|').split('|'):
                cell=cell.strip()
                if cell.startswith('**') and cell.endswith('**'):
                    cell=cell[2:-2]
                row.append(cell)
            if len(row)!=width:
                issues.append(f'TABLE {prefix}: {len(row)} columns, need {width}');continue
            rows.append(row)
        if not rows:issues.append(f'TABLE {prefix}: no data rows')
        return rows
    def schema(rows,expected,label,keys=None):
        got=[r[0] for r in rows] if keys is None else list(keys)
        counts=collections.Counter(got)
        if set(got)!=set(expected) or any(n!=1 for n in counts.values()):
            issues.append(f'{label}: wrong row IDs; missing={sorted(set(expected)-set(got))}, extra={sorted(set(got)-set(expected))}, duplicates={[k for k,n in counts.items() if n!=1]}')
    def eq(label,text,value,fmt):
        want=format(value,fmt)
        if text!=want:issues.append(f'{label}: {text!r} != {want!r}')
    def cells(label,text,values,formats):
        parts=[x.strip() for x in text.split('/')]
        if len(parts)!=len(values):issues.append(f'{label}: {len(parts)} subcells, need {len(values)}');return
        for i,(got,value,fmt) in enumerate(zip(parts,values,formats)):eq(f'{label}[{i}]',got,value,fmt)

    sp={r['session']:r for r in load('structure_probe.json')['A']}
    expected=[r['session'] for r in sorted(sp.values(),key=lambda x:x['z'])[:9]]
    rows=table('| session | fields | good share',7);schema(rows,expected,'T1')
    for r in rows:
        if r[0] not in sp:continue
        d=sp[r[0]]
        for i,key,fmt in [(1,'n','d'),(2,'good_frac','.2f'),(3,'runs','d'),(4,'expected','.1f'),(5,'z','.2f')]:eq('T1 '+r[0]+' '+key,r[i],d[key],fmt)
        eq('T1 p '+r[0],r[6],'<1e-6' if d['p']<1e-6 else f"{d['p']:.4f}",'s')

    ls=load('label_sufficiency.json')
    for prefix,key,cell_type in [('| culture age','single_image_agreement_by_day',False),('| cell line | fields | single','single_image_agreement_by_cell',True)]:
        data=ls[key];expected={k.replace('cell_type_','') if cell_type else k for k in data}
        rows=table(prefix,3);schema(rows,expected,'T2')
        for r in rows:
            name='cell_type_'+r[0] if cell_type else r[0]
            if name not in data:continue
            eq('T2 n '+r[0],r[1],data[name]['n'],'d');eq('T2 agreement '+r[0],r[2],data[name]['agree'],'.3f')

    for label,prefix,name,keys in [
        ('T3','| budget k | random: acc / recall / err','adaptive_sampling.json',['chip_acc','bad_recall','frac_err']),
        ('T4','| budget k | random: acc / recall |','policy_model_in_loop.json',['chip_acc','bad_recall']),
        ('T5','| budget k | mean: acc / sens / spec','aggregation_results.json',['acc','sens','spec'])]:
        data=load(name)
        if label=='T5':data=data['summary']
        rows=table(prefix,5);schema(rows,data,label)
        policies=['mean','max','top2'] if label=='T5' else ['random','window','adaptive']
        for r in rows:
            if r[0] not in data:continue
            for i,policy in enumerate(policies,1):cells(f'{label} k={r[0]} {policy}',r[i],[data[r[0]][policy][k] for k in keys],['.3f']*len(keys))
            eq(label+' majority '+r[0],r[4],data[r[0]]['majority_share'],'.3f')

    rc=load('recovery_test.json')
    mappings={'run length only':'length only','artifact signals only':'artifact signals only','run length + artifacts':'length + artifact','all features':'all','all features except *reaches the end*':'all but reaches_end'}
    rows=table('| feature set | CV AUC',2);schema(rows,list(mappings)+['permutation null','simple rule (length ≤ 2 ⇒ recover)'],'Recovery')
    for r in rows:
        if r[0] in mappings:eq('Recovery '+r[0],r[1],rc['cv_auc'][mappings[r[0]]],'.3f')
        elif r[0]=='permutation null':eq('Recovery null',r[1],f"mean {rc['null_mean']:.3f}, p95 **{rc['null_p95']:.3f}**",'s')
        elif r[0]=='simple rule (length ≤ 2 ⇒ recover)':eq('Recovery rule',r[1],f"accuracy {rc['simple_rule_acc']:.3f} vs base rate **{rc['recovery_rate']:.3f}**",'s')

    ic=load('inner_cv_minfields.json');rows=table('| on 68 non-test chips',5)
    schema(rows,['chip accuracy, every chip called','confident-but-wrong chips','fields per chip'],'InnerCV')
    by={r[0]:r for r in rows}
    for i,k in enumerate(['1','8','12'],1):
        d=ic['by_min_fields'][k]
        for name,key,fmt in [('chip accuracy, every chip called','force_acc','.3f'),('confident-but-wrong chips','false_confident','d'),('fields per chip','mean_fields','.1f')]:
            if name in by:eq('InnerCV '+name+' min'+k,by[name][i],d[key],fmt)
    if 'chip accuracy, every chip called' in by:eq('InnerCV all accuracy',by['chip accuracy, every chip called'][4],ic['all_fields']['force_acc'],'.3f')
    if 'fields per chip' in by:eq('InnerCV all fields',by['fields per chip'][4],ic['all_fields']['mean_fields'],'.1f')
    if 'confident-but-wrong chips' in by:eq('InnerCV unavailable',by['confident-but-wrong chips'][4],'—','s')

    tool=load('tool_evaluation.json');eff=load('efficiency.json');seq=tool['chip_sequential_by_min_fields']['8']
    rows=table('| metric | value |',2)
    labels=['per-field accuracy / AUC','chip accuracy, all fields, mean','chip accuracy among called chips','false-confident calls','inconclusive (budget exhausted near P = 0.5)','mean fields used']
    keys=[next((name for name in labels if r[0].replace('**','').startswith(name)),r[0]) for r in rows];schema(rows,labels,'ToolMetrics',keys)
    for name,r in zip(keys,rows):
        if name==labels[2]:eq('ToolMetrics called label',r[0],'**chip accuracy among called chips** (spread fields, min 8 — tuned on these chips, §6.2(c))','s')
        if name==labels[3]:eq('ToolMetrics confident definition',r[0],f"**false-confident calls** (conf ≥ 0.9 and wrong, of {seq['n_confident']} calls)",'s')
        got=r[1].strip('*'); clean=lambda s:s.replace('**','')
        if name==labels[0]:want=None  # computed from saved labels below
        elif name==labels[1]:want=f"{tool['chip_all_fields_mean_acc']:.2f} (calling every chip *pass*: {load('vs_all_pass.json')['test_min8']['all_pass_acc']:.2f}, {round(load('vs_all_pass.json')['test_min8']['all_pass_acc']*tool['n_chips'])} of {tool['n_chips']})"
        elif name==labels[2]:want=f"{seq['chip_acc_among_confident']:.3f} (95% Wilson CI {tool['chip_acc_among_confident_wilson95'][0]:.2f}–{tool['chip_acc_among_confident_wilson95'][1]:.2f}, n={seq['n_confident']})"
        elif name==labels[3]:want=f"{100*seq['false_confident_rate']:.0f}% ({round(seq['false_confident_rate']*seq['n_confident'])}/{seq['n_confident']})"
        elif name==labels[4]:want=f"{100*(1-seq['n_confident']/tool['n_chips']):.0f}%"
        elif name==labels[5]:want=f"{seq['mean_fields']:.1f} (a plain cap of 12 spread fields: {eff['cap_k']['12']['fields']:.1f} fields, {eff['cap_k']['12']['acc']:.2f})"
        else:want=None
        if name==labels[0]:
            preds=load('perfield_preds_384_s0.json')['test'];majority=sum(y==1 for y in preds['y'])/len(preds['y'])
            want=f"{tool['field_accuracy']:.3f} / {tool['field_auc']:.3f} (always predicting `good`: {majority:.3f})"
        if want is not None:eq('ToolMetrics '+name,clean(got),clean(want),'s')

    rows=table('| policy | fields per chip',4)
    wanted={'all fields (mean of every field)':(eff['mean_fields_per_chip'],tool['n_chips'],eff['all_fields_acc']),
            'fixed k = 12, spread (chips with ≥ 12 fields only)':(eff['fixed_k']['12']['fields'],eff['fixed_k']['12']['n'],eff['fixed_k']['12']['acc']),
            'cap of 12 spread fields (all fields if fewer)':(eff['cap_k']['12']['fields'],tool['n_chips'],eff['cap_k']['12']['acc']),
            '**sequential, force** (min 8)':(eff['sequential']['force_min8']['fields'],tool['n_chips'],eff['sequential']['force_min8']['acc']),
            '**sequential, abstain** (min 8)':(eff['sequential']['abstain_min8']['fields'],eff['sequential']['abstain_min8']['n_called'],eff['sequential']['abstain_min8']['acc'])}
    schema(rows,wanted,'T6')
    for r in rows:
        if r[0] not in wanted:continue
        fields,n,acc=wanted[r[0]];eq('T6 fields '+r[0],r[1],fields,'.1f');eq('T6 called '+r[0],r[2],f'{n}/{tool["n_chips"]}','s');eq('T6 acc '+r[0],r[3],acc,'.3f')

    pc=load('per_chip_calls.json')['per_session'];rows=table('| chip | fields | bad share | field acc |',9);ids=[]
    for rr in rows:
        eq('T7 spacer',rr[4],'','s')
        for r in [rr[:4],rr[5:]]:
            if not r[0]:
                if any(r):issues.append('T7 partially empty slot')
                continue
            ids.append(r[0]);d=pc.get(r[0])
            if d is None:continue
            eq('T7 n '+r[0],r[1],d['n_fields'],'d');eq('T7 bad '+r[0],r[2],f"{100*d['bad_share']:.0f}%",'s');eq('T7 acc '+r[0],r[3],d['field_acc'],'.3f')
    schema(rows,pc,'T7',ids)

    lo=load('lolo_cellline.json')['per_cell'];rows=table('| held-out cell line',5)
    names=[k.replace('cell_type_','') for k in lo]+['mean of the six','in-distribution (all lines seen, §6.3)'];schema(rows,names,'T8')
    for r in rows:
        d=lo.get('cell_type_'+r[0])
        if d:
            eq('T8 size '+r[0],r[1],f"{d['n_test']} ({d['n_sessions_test']})",'s');eq('T8 acc '+r[0],r[2],f"{d['acc']:.3f} ({d['majority_share']:.3f})",'s')
            eq('T8 balanced '+r[0],r[3],d['bal_acc'],'.3f');eq('T8 auc '+r[0],r[4],d['auc'],'.3f')
        elif r[0]=='mean of the six':
            eq('T8 aggregate counts',r[1],f"{sum(d['n_test'] for d in lo.values()):,} ({sum(d['n_sessions_test'] for d in lo.values())})",'s')
            for i,k in [(2,'acc'),(3,'bal_acc'),(4,'auc')]:eq('T8 mean '+k,r[i],sum(d[k] for d in lo.values())/len(lo),'.3f')
        elif r[0]=='in-distribution (all lines seen, §6.3)':
            eq('T8 baseline counts',r[1],f"{tool['n_fields']} ({tool['n_chips']})",'s')
            eq('T8 baseline acc',r[2],tool['field_accuracy'],'.3f');eq('T8 baseline auc',r[4],tool['field_auc'],'.3f')
            preds=load('perfield_preds_384_s0.json')['test'];p=preds['p'];y=preds['y']
            rates=[sum((v>.5)==(lab==0) for v,lab in zip(p,y) if lab==c)/sum(lab==c for lab in y) for c in [0,1]]
            eq('T8 baseline balanced',r[3],sum(rates)/2,'.3f')
    return issues


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--here',type=Path,default=HERE);args=ap.parse_args()
    try:issues=check_tables(here=args.here)
    except (ValueError,KeyError,IndexError,TypeError) as error:issues=[f'Table schema/result error: {error}']
    for issue in issues:print('MISMATCH',issue)
    print(f'{len(issues)} table mismatches')
    raise SystemExit(bool(issues))


if __name__=='__main__':main()
