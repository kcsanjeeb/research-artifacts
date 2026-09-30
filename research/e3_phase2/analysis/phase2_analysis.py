#!/usr/bin/env python3
"""DECAF Phase 2 analysis payload (run on Ada after arms complete).

Inputs (run dir /data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB):
  analyze_bias.jsonl           per-block write-time signals (encode-only arm)
  <arm>/commit_log.jsonl       per-question commit/entropy/materialization logs
  judged_<arm>_72b.json        72B judge scores per arm
  /data3/.../code/decaf/data/rvs/ego/ego4d_oe.json  (video durations for GB/h)

Outputs: analysis/phase2_analysis.json + printed summary.
"""
import json, math, os, sys
from collections import defaultdict

RUN = sys.argv[1] if len(sys.argv) > 1 else '/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB'
ANNO = '/data3/zhuotaotian2_e2/code/decaf/data/rvs/ego/ego4d_oe.json'
ARMS = ['answers_7b_b_deferred', 'answers_7b_b_writetime', 'answers_7b_b_writetime_rawbias']
out = {}

# ---- 1. write-time bias analysis (C1 thesis evidence) -----------------------
ap = os.path.join(RUN, 'analyze_bias.jsonl')
bias = {}
if os.path.exists(ap):
    raw, deb = [], []
    nblocks = 0
    last_bytes = 0
    with open(ap) as f:
        for line in f:
            r = json.loads(line)
            raw.append(r['attn_raw']); deb.append(r['attn_deb'])
            nblocks = r['block']; last_bytes = r['store_bytes_total']
    import numpy as np
    raw, deb = np.array(raw), np.array(deb)
    res_raw = float(np.abs(raw - raw.mean(0)).mean())
    res_deb = float(np.abs(deb - deb.mean(0)).mean())
    # positional bias curve L1 norm (systematic component vs position)
    curve_raw = raw.mean(0); curve_deb = deb.mean(0)
    bias = {
        'n_blocks': int(len(raw)),
        'mean_abs_residual_raw': res_raw,
        'mean_abs_residual_debiased': res_deb,
        'reduction_pct': 100 * (1 - res_deb / res_raw),
        'pos_curve_raw': curve_raw.round(6).tolist(),
        'pos_curve_debiased': curve_deb.round(6).tolist(),
    }
out['C1_bias'] = bias

# ---- 2. per-arm commit/entropy/materialization stats ------------------------
anno = json.load(open(ANNO))
dur = {}
for v in anno:
    dur[v['video_id']] = max(c['end_time'] for c in v['conversations']) / 3600.0  # hours
total_hours = sum(dur.values())

arms = {}
for arm in ARMS:
    p = os.path.join(RUN, arm, 'commit_log.jsonl')
    if not os.path.exists(p):
        continue
    recs = [json.loads(l) for l in open(p)]
    n_pass2 = sum(1 for r in recs if 'pass2' in r)
    ents = [r['pass1']['mean_entropy'] for r in recs if 'pass1' in r]
    mat1 = [r['pass1'].get('committed_slots', 0) for r in recs if 'pass1' in r]
    sb = defaultdict(int)
    for r in recs:
        if 'store_bytes' in r:
            sb[r['video_id']] = max(sb[r['video_id']], r['store_bytes']['store_bytes_total'])
    gbh = sum(sb.values()) / total_hours / 1e9 if total_hours else None
    arms[arm] = {
        'n_questions': len(recs),
        'pass2_rate': n_pass2 / max(1, len(recs)),
        'mean_entropy_pass1': sum(ents) / max(1, len(ents)),
        'mean_committed_slots': sum(mat1) / max(1, len(mat1)),
        'store_gb_per_h': gbh,
        'store_bytes_by_video': dict(sb),
    }
    jp = os.path.join(RUN, f'judged_{arm}_72b.json')
    if os.path.exists(jp):
        j = json.load(open(jp))
        arms[arm]['judge'] = {'acc': j['accuracy'], 'score': j['average_score']}
out['arms'] = arms

# ---- 3. two-stage budget formulation summary (numbers layer) ----------------
d = arms.get('answers_7b_b_deferred', {})
w = arms.get('answers_7b_b_writetime', {})
formulation = {
    'stage1_write': ('per-segment retention under tax B: writetime = single ranked '
                     'keep-list (committed); deferred = union of per-signal '
                     'keep-lists B/2 each (hedged superset, uncommitted)'),
    'stage2_query': ('per-query materialization of <=64 frame slots: retrieval on '
                     'content keys -> cross-grain consistency rerank (gamma) -> '
                     'coarse-to-fine commitment; uncertainty (answer entropy) '
                     'triggers finer re-commitment'),
    'regret': 'PENDING oracle grain-policy runs (seg/frame/patch) for the clairvoyant bound',
}
if d.get('judge') and w.get('judge'):
    formulation['delta_acc_deferred_minus_writetime'] = (
        d['judge']['acc'] - w['judge']['acc']) * 100
    formulation['same_store_note'] = (
        'identical hedged superset; writetime commits keep-list at write time, '
        'deferred commits per query')
out['two_stage'] = formulation

os.makedirs(os.path.join(RUN, 'analysis'), exist_ok=True)
with open(os.path.join(RUN, 'analysis', 'phase2_analysis.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2)[:4000])
