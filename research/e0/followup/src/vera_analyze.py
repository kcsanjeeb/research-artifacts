"""VERA (InternVL2-8B) analysis on the two E0.7 s0 cells: same isotonic
crossfit-by-video-parity fusion, same per-cell oracle ceilings, same capture
fraction as e07.py. score_b = P("1") logprob at the label position (primary,
as E0.6); score_c = VERA's official hard parse (fused raw, as VERA itself
reports). Output: vera_e07.json.
"""
import glob
import json
import os
import sys

import numpy as np

E0 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(E0, 'src'))
import common  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlmvalue import auc_rank, ap_score, event_stats  # noqa: E402
from vlm_calibrated import pava, iso_apply  # noqa: E402
from e07 import load_scores_labels, TAU  # noqa: E402

OUT = os.path.join(E0, 'followup', 'results')
SNIP, FPS = 16, 30.0


def load_vera(gate):
    vlm = {}
    for f in sorted(glob.glob(os.path.join(OUT, f'vera_scores_{gate}_s0_shard*.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            vlm[(r['vi'], r['si'])] = r
    return vlm


def main():
    ids, scores, lens, labels = load_scores_labels()
    gt_frames = common.load_frame_gt()

    def full_frame(fused):
        concat = np.concatenate([np.repeat(fused[i, :lens[i]], SNIP)
                                 for i in range(len(ids))])
        n = min(len(concat), len(gt_frames))
        return concat[:n], gt_frames[:n]

    fs0, g0 = full_frame(scores)
    auc_t1, ap_t1 = auc_rank(fs0, g0), ap_score(fs0, g0)
    print(f"baseline tier-1: AUC={auc_t1:.4f} AP={ap_t1:.4f}")

    cells = []
    for gate in ('band', 'topk'):
        d = np.load(os.path.join(OUT, 'escalated_clips.npz' if gate == 'band'
                                 else 'escalated_topk.npz'), allow_pickle=True)
        vis, sis = d['vi'], d['si']
        vlm = load_vera(gate)
        assert len(vlm) == len(vis), f"{gate}: {len(vlm)} != {len(vis)}"
        gt_pool = labels[vis, sis]
        s_pool = scores[vis, sis]
        # per-cell ceiling
        fused = scores.copy()
        fused[vis, sis] = labels[vis, sis]
        fs, g = full_frame(fused)
        ceiling = auc_rank(fs, g) - auc_t1
        pv = np.array([vlm[(vi, si)]['score_b'] for vi, si in zip(vis, sis)])
        pc = np.array([vlm[(vi, si)]['score_c'] for vi, si in zip(vis, sis)])
        n_none = sum(1 for (vi, si) in zip(vis, sis)
                     if vlm[(vi, si)]['score_b'] is None)
        print(f"[{gate}] pool={len(vis)} anomalous={gt_pool.mean():.3f} "
              f"ceiling={ceiling:+.4f} score_b missing={n_none}")
        pv_filled = np.where(np.isnan(pv.astype(float)) | (pv == None), 0.0,
                             pv.astype(float))
        # isotonic crossfit by video parity on score_b
        cal = {}
        for parity in (0, 1):
            fit_m = (vis % 2) == parity
            app_m = ~fit_m
            kx, ky = pava(pv_filled[fit_m], gt_pool[fit_m])
            vals = iso_apply(kx, ky, pv_filled[app_m])
            for idx, v in zip(np.flatnonzero(app_m), vals):
                cal[idx] = float(v)
        cell = {'gate': gate, 'span_snippets': 0, 'model': 'VERA/InternVL2-8B',
                'oracle_ceiling_dAUC': float(ceiling)}
        for name, score_map in (
                ('replace_b_isotonic', {i: cal[i] for i in range(len(vis))}),
                ('replace_c_raw', {i: float(pc[i]) for i in range(len(vis))})):
            fused = scores.copy()
            for idx, (vi, si) in enumerate(zip(vis, sis)):
                fused[vi, si] = score_map[idx]
            fs, g = full_frame(fused)
            d = auc_rank(fs, g) - auc_t1
            cell[name] = {'dAUC': float(d), 'dAP': float(ap_score(fs, g) - ap_t1),
                          'capture_fraction': float(d / ceiling)}
            print(f"  {name}: dAUC={d:+.4f} capture={d/ceiling:+.1%}")
        cell['within_pool_auc_vlm'] = auc_rank(pv_filled, gt_pool)
        cell['within_pool_auc_tier1'] = auc_rank(s_pool, gt_pool)
        cell['vlm_yes_rate'] = float((pv_filled > 0.5).mean())
        cell['vlm_acc'] = float(((pv_filled > 0.5).astype(int) == gt_pool).mean())
        cell['const_majority_acc'] = float(max(gt_pool.mean(), 1 - gt_pool.mean()))
        cell['agreement_c_vs_b@0.5'] = float(((pv_filled > 0.5).astype(int) == pc).mean())
        cells.append(cell)
        print(f"  poolAUC vlm={cell['within_pool_auc_vlm']:.3f} "
              f"t1={cell['within_pool_auc_tier1']:.3f} "
              f"acc={cell['vlm_acc']:.3f}/const={cell['const_majority_acc']:.3f} "
              f"yes={cell['vlm_yes_rate']:.3f} "
              f"c_vs_b_agree={cell['agreement_c_vs_b@0.5']:.3f}")

    doc = {'config': {'model': 'OpenGVLab/InternVL2-8B', 'prompt': 'VERA learned '
                      'prompt + 5 guiding questions (verbatim from repo)',
                      'clip_spec': 'same as E0.7 cells (12 frames, 448x448, s0)',
                      'fusion': 'isotonic_crossfit_by_video_parity',
                      'note': 'VERA natively uses 8 frames over a 10s window and '
                              'a hard 0/1 parse; we hold the grid clip spec and '
                              'report both the logprob score (primary) and '
                              'VERA\'s own parse'},
           'baseline': {'auc_tier1': auc_t1, 'ap_tier1': ap_t1},
           'cells': cells}
    with open(os.path.join(OUT, 'vera_e07.json'), 'w') as f:
        json.dump(doc, f, indent=2)
    print(f"[saved] {OUT}/vera_e07.json")


if __name__ == '__main__':
    main()
