#!/usr/bin/env python3
'''W4.1-corrected multiple-comparisons family (senior directive 4, 2026-09-30).

Identical methodology to w31_multiple_comparisons.py, but the writetime arm in
family A (and the writetime filter-in for family B) is the W4.1 TRUE-0.5-fps run
(runs/20260930_0935_w41_fpsfix/answers_7b_writetime_05fps), replacing the old
0.25fps-effective writetime entries (runs/20260928_0315_e3_phase2_compB).
Everything else (arms, subsets, corrections) is unchanged. Originals left intact;
outputs go to analysis/w41_family_correction/.
'''
import json, csv, itertools, os
from scipy import stats

B = '/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB/'
P0 = '/data3/zhuotaotian2_e2/runs/20260927_0000_e3_phase0/'
W25 = '/data3/zhuotaotian2_e2/runs/20260926_1455_w25_judge_harness/'
W41 = '/data3/zhuotaotian2_e2/runs/20260930_0935_w41_fpsfix/'
OUT = '/data3/zhuotaotian2_e2/deliverables/e3_phase2/analysis/w41_family_correction/'
os.makedirs(OUT, exist_ok=True)

FAMILY_A = {
    'deferred':            B + 'judged_answers_7b_b_deferred_72b.json',
    'writetime_05fps':     W41 + 'judged_7b_writetime_05fps_72b.json',
    'writetime_rawbias':   B + 'judged_answers_7b_b_writetime_rawbias_72b.json',
    'writetime_tax025':    B + 'judged_answers_7b_b_writetime_tax025_72b.json',
    'writetime_tax075':    B + 'judged_answers_7b_b_writetime_tax075_72b.json',
    'mukv_paper':          P0 + 'judged_7b_paper_72b.json',
    'stock_rekv_w25':      W25 + 'judged_7b_72b.json',
}
ORACLE = {
    'oracle_seg':    B + 'judged_oracle_seg_72b.json',
    'oracle_frame':  B + 'judged_oracle_frame_72b.json',
    'oracle_patch':  B + 'judged_oracle_patch_72b.json',
    'oracle_default':B + 'judged_oracle_default_72b.json',
}
SUBSET_VIDEOS = {v['video_id'] for v in json.load(open(B + 'oracle_subset_anno.json'))}

def load_ordered(path, subset=None):
    recs = json.load(open(path))['judged']
    seen, out = {}, []
    for r in recs:
        if subset is not None and r['video_id'] not in subset:
            continue
        k = (r['video_id'], r['question'])
        i = seen.get(k, 0)
        seen[k] = i + 1
        out.append(((k, i), 1 if r['judge_pred'] == 'yes' else 0))
    return out

def mcnemar(a_map, b_map):
    keys = sorted(set(a_map) & set(b_map))
    ab = ba = both = bw = 0
    for k in keys:
        x, y = a_map[k], b_map[k]
        if x == 1 and y == 1: both += 1
        elif x == 0 and y == 0: bw += 1
        elif x == 1: ab += 1
        else: ba += 1
    n = len(keys); d = ab + ba
    if d > 0:
        p_exact = stats.binomtest(ab, d, 0.5).pvalue
        chi2 = (abs(ab - ba) - 1) ** 2 / d
        p_cc = stats.chi2.sf(chi2, 1)
    else:
        p_exact = p_cc = 1.0
    return dict(n=n, both_correct=both, both_wrong=bw, a_right_b_wrong=ab,
                b_right_a_wrong=ba, p_cc=p_cc, p_exact=p_exact,
                acc_a=(both + ab) / n, acc_b=(both + ba) / n, diff=(ab - ba) / n)

def all_pairs(arms):
    res = {}
    for a, b in itertools.combinations(sorted(arms), 2):
        res[f'{a} vs {b}'] = mcnemar(dict(arms[a]), dict(arms[b]))
    return res

def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj

def bh(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = min(1.0, m / (rank + 1) * pvals[i])
        running = min(running, val)
        adj[i] = running
    return adj

armsA = {k: load_ordered(v) for k, v in FAMILY_A.items()}
pairsA = all_pairs(armsA)
json.dump(pairsA, open(OUT + 'w41_family_A.json', 'w'), indent=1)

armsB = {k: load_ordered(v) for k, v in ORACLE.items()}
armsB['deferred'] = load_ordered(FAMILY_A['deferred'], SUBSET_VIDEOS)
armsB['writetime_05fps'] = load_ordered(FAMILY_A['writetime_05fps'], SUBSET_VIDEOS)
pairsB = all_pairs(armsB)
json.dump(pairsB, open(OUT + 'w41_family_B.json', 'w'), indent=1)

accA = {k: sum(c for _, c in v) / len(v) for k, v in armsA.items()}
accB = {k: sum(c for _, c in v) / len(v) for k, v in armsB.items()}

# corrected-arm comparisons of interest
KEY = ['mukv_paper vs writetime_05fps', 'stock_rekv_w25 vs writetime_05fps']
res = {'replaced_arm': 'writetime (old 0.25fps-effective) -> writetime_05fps (W4.1 true 0.5fps)',
       'corrected_comparisons': {}, 'families': {}}
for pkey in ('p_exact', 'p_cc'):
    res['corrected_comparisons'][pkey] = {k: pairsA[k][pkey] for k in KEY}
for fam, pairs in (('A(m=21, full-run RVS arms)', pairsA),
                   ('B(m=15, oracle subset arms)', pairsB),
                   ('union(m=36)', {**pairsA, **{f'B::{k}': v for k, v in pairsB.items()}})):
    for pkey in ('p_exact', 'p_cc'):
        names = list(pairs)
        p = [pairs[n][pkey] for n in names]
        h = holm(p); f = bh(p)
        corr = {n: (h[i], f[i]) for i, n in enumerate(names)}
        entry = {'m': len(pairs),
                 'min_holm': min(h), 'min_bh': min(f),
                 'any_holm_lt_0.05': any(v < 0.05 for v in h),
                 'any_bh_lt_0.05': any(v < 0.05 for v in f)}
        for k in KEY:
            if k in pairs:
                entry.setdefault('corrected_comparisons', {})[k] = {
                    'raw': pairs[k][pkey], 'holm': corr[k][0], 'bh': corr[k][1]}
        res['families'][f'{fam}|{pkey}'] = entry
json.dump(res, open(OUT + 'w41_correction.json', 'w'), indent=1)

with open(OUT + 'w41_matrix.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['family', 'pair', 'n', 'both_correct', 'both_wrong',
                'a_right_b_wrong', 'b_right_a_wrong', 'diff_pt',
                'p_exact', 'p_cc', 'holm_exact', 'bh_exact'])
    for fam, pairs in (('A', pairsA), ('B', pairsB)):
        names = list(pairs)
        p = [pairs[n]['p_exact'] for n in names]
        h = holm(p); f = bh(p)
        corr = {n: (h[i], f[i]) for i, n in enumerate(names)}
        for name, s in sorted(pairs.items(), key=lambda kv: kv[1]['p_exact']):
            w.writerow([fam, name, s['n'], s['both_correct'], s['both_wrong'],
                        s['a_right_b_wrong'], s['b_right_a_wrong'],
                        round(100 * s['diff'], 2), round(s['p_exact'], 4),
                        round(s['p_cc'], 4), round(corr[name][0], 4), round(corr[name][1], 4)])

print('== Family A arm accuracies (n=1465) ==')
for k, v in sorted(accA.items()):
    print(f'  {k:20s} {100*v:.2f}')
print('== Family B arm accuracies (n=316 subset) ==')
for k, v in sorted(accB.items()):
    print(f'  {k:20s} {100*v:.2f}')
print('\n== Family A: 21 pairs by p_exact ==')
for name, s in sorted(pairsA.items(), key=lambda kv: kv[1]['p_exact']):
    print(f'  {name:42s} diff {100*s["diff"]:+5.2f}  b={s["a_right_b_wrong"]:3d} c={s["b_right_a_wrong"]:3d}  p={s["p_exact"]:.4f}')
print('\n== Corrections ==')
print(json.dumps(res, indent=1))
