#!/usr/bin/env python
"""Inter-judge agreement between two judged jsons (same question order)."""
import json, sys

a = json.load(open(sys.argv[1]))["judged"]
b = json.load(open(sys.argv[2]))["judged"]
n = min(len(a), len(b))
both_yes = both_no = disc = 0
scores_a, scores_b = [], []
for x, y in zip(a[:n], b[:n]):
    assert x["question"] == y["question"]
    pa, pb = x["judge_pred"], y["judge_pred"]
    if pa and pb:
        if pa == pb: 
            both_yes += (pa == "yes"); both_no += (pa == "no")
        else: disc += 1
    sa, sb = x["judge_score"], y["judge_score"]
    if sa is not None and sb is not None:
        scores_a.append(sa); scores_b.append(sb)
agree = (both_yes + both_no) / max(1, both_yes + both_no + disc)
mean_abs = sum(abs(x - y) for x, y in zip(scores_a, scores_b)) / max(1, len(scores_a))
print(f"n={n} agree={agree*100:.1f}% (yes={both_yes} no={both_no} discordant={disc}) score_MAE={mean_abs:.2f} n_scored_both={len(scores_a)}")
