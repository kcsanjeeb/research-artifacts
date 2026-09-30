#!/usr/bin/env python
"""DECAF Gate 1 verdict.

Gate (beat-mukv.md §5 Phase 1): retrieval recall +>=5 pts OR RVS-Ego Acc +>=2 pts
(72B-judged) for the position-free store vs the entangled store at equal memory.

Inputs: judged DECAF jsons, judged W2.5 entangled jsons, recall summaries.
Output: gate1_verdict.txt + gate1_verdict.json
"""
import json
import sys


def load(p):
    with open(p) as f:
        return json.load(f)


def judged_metrics(path):
    d = load(path)
    # w25_judge.py output: {'n':..., 'acc':..., 'score':...} (acc in percent)
    if isinstance(d, dict) and 'acc' in d:
        return {'n': d['n'], 'acc': d['acc'], 'score': d['score']}
    raise ValueError(f'unrecognized judged format: {path}')


def recall_metrics(path):
    d = load(path)
    return d


def fmt_pct(x):
    return f'{x:+.2f} pts' if x is not None else 'n/a'


def main():
    (judged_decaf_7b, judged_decaf_05b,
     judged_ent_7b, judged_ent_05b,
     recall_decaf_7b, recall_decaf_05b,
     recall_ent_7b, recall_ent_05b, out_prefix) = sys.argv[1:10]

    d7 = judged_metrics(judged_decaf_7b)
    d5 = judged_metrics(judged_decaf_05b)
    e7 = judged_metrics(judged_ent_7b)
    e5 = judged_metrics(judged_ent_05b)

    r_d7 = recall_metrics(recall_decaf_7b)
    r_d5 = recall_metrics(recall_decaf_05b)
    r_e7 = recall_metrics(recall_ent_7b)
    r_e5 = recall_metrics(recall_ent_05b)

    lines = []
    lines.append('DECAF Gate 1 verdict — position-disentangled store vs entangled store (equal memory)')
    lines.append('')
    lines.append('RVS-Ego accuracy (72B-judged, n=1465):')
    for tag, d, e in [('7B', d7, e7), ('0.5B', d5, e5)]:
        lines.append(f'  {tag}: entangled Acc {e["acc"]} / Score {e["score"]}  |  '
                     f'position-free Acc {d["acc"]} / Score {d["score"]}  |  '
                     f'delta Acc {fmt_pct(d["acc"] - e["acc"])} / Score {fmt_pct(d["score"] - e["score"])}')
    lines.append('')
    lines.append('Retrieval recall vs temporal oracle (layer-0 retrieved blocks, strict window;')
    lines.append('oracle = frames in [start,end] of ego4d_oe.json at 0.5 fps, restricted to encoded frames):')
    for tag, rd, re_ in [('7B', r_d7, r_e7), ('0.5B', r_d5, r_e5)]:
        for w in ['strict', 'relaxed_pm30s']:
            dm = rd[w]['micro_recall_pct']
            em = re_[w]['micro_recall_pct']
            lines.append(f'  {tag} {w}: entangled micro {em}%  |  position-free micro {dm}%  |  '
                         f'delta {fmt_pct((dm - em) if (dm is not None and em is not None) else None)}'
                         f'  (n={rd[w]["n_questions"]})')
    lines.append('')

    d_acc_7b = d7['acc'] - e7['acc']
    d_acc_05b = d5['acc'] - e5['acc']
    dr7 = r_d7['strict']['micro_recall_pct'] - r_e7['strict']['micro_recall_pct'] \
        if None not in (r_d7['strict']['micro_recall_pct'], r_e7['strict']['micro_recall_pct']) else None
    dr5 = r_d5['strict']['micro_recall_pct'] - r_e5['strict']['micro_recall_pct'] \
        if None not in (r_d5['strict']['micro_recall_pct'], r_e5['strict']['micro_recall_pct']) else None

    passed = (dr7 is not None and dr7 >= 5.0) or (dr5 is not None and dr5 >= 5.0) \
        or d_acc_7b >= 2.0 or d_acc_05b >= 2.0
    lines.append(f'Gate criteria: recall +>=5 pts OR Acc +>=2 pts (primary: 7B).')
    lines.append(f'  recall delta 7B: {fmt_pct(dr7)} | 0.5B: {fmt_pct(dr5)}')
    lines.append(f'  acc delta    7B: {fmt_pct(d_acc_7b)} | 0.5B: {fmt_pct(d_acc_05b)}')
    lines.append(f'GATE 1 VERDICT: {"PASS" if passed else "FAIL"}')
    verdict = {
        'gate1': 'PASS' if passed else 'FAIL',
        'acc_delta_7b': d_acc_7b, 'acc_delta_05b': d_acc_05b,
        'recall_delta_7b': dr7, 'recall_delta_05b': dr5,
        'entangled': {'7b': e7, '0.5b': e5},
        'position_free': {'7b': d7, '0.5b': d5},
    }
    with open(out_prefix + '.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(out_prefix + '.json', 'w') as f:
        json.dump(verdict, f, indent=2)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
