"""T1 complement on our corpus (no GPU): retrieval frequency f_j and marginal
value v_j (balanced-acc drop under leave-one-out) for each of the 749 FleetMem
memory events over the 220 UCF/XD SMB queries, in the v0.1 final query config
(dual-channel embeddings + tag boost + refined-combo temporal, tau=0.48,
verify=none). Value per byte = v_j / bytes_j (memory record JSON bytes + 1536B
embedding). Gini + Lorenz + uniform-vs-oracle retention curves.

Usage: python t1_smb.py --run_dir R
"""
import argparse
import json
import os
import sys

import numpy as np

FLEETMEM = os.path.expanduser("~/e1/fleetmem")
sys.path.insert(0, FLEETMEM)
from querypath import load_memory, load_jsonl, span_iou, evaluate  # noqa: E402
from querypath_v01 import query_category  # noqa: E402

INGEST = os.path.expanduser("~/e1/runs/20260918_0111_writepath_fullcorpus")
TAGRUN = os.path.expanduser("~/e1/runs/20260918_0802_v01_tagging")
SMB = os.path.expanduser("~/e1/smb")
TAU = 0.48
TOP_R = 3

EMB_BYTES = 384 * 4  # fp32 embedding


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.run_dir, "artifacts"), exist_ok=True)

    recs, E = load_memory(INGEST)
    E2 = np.load(os.path.join(INGEST, "query_eval", "text_embs.npy"))
    tags = {}
    import glob
    for p in glob.glob(os.path.join(TAGRUN, "artifacts", "tags_*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            tags[r["event_id"]] = r.get("category_tag")
    refined = json.load(open(os.path.join(TAGRUN, "refined_spans.json")))
    queries = [q for q in load_jsonl(os.path.join(SMB, "queries.jsonl"))
               if q["dataset"] in ("ucf", "xd")]
    n_ev, n_q = len(recs), len(queries)
    print(f"{n_ev} events, {n_q} queries", flush=True)

    from querypath import QueryEmbedder
    emb = QueryEmbedder("cpu")
    q_emb = emb.embed([q["text"] for q in queries])
    qcats = [query_category(q["text"], q["dataset"]) for q in queries]

    def score_matrix(mask=None):
        S = np.maximum(q_emb @ E.T, q_emb @ E2.T)
        for qi, c in enumerate(qcats):
            if not c:
                continue
            for j in range(n_ev):
                if tags.get(recs[j]["L0"]["event_id"]) == c:
                    S[qi, j] = max(S[qi, j], 0.75)
        if mask is not None:
            S[:, mask] = -1e9
        return S

    def run_eval(mask=None):
        S = score_matrix(mask)
        results = []
        for qi, q in enumerate(queries):
            typ = q["type"]
            vid = q["relevant_videos"][0] if q["relevant_videos"] else None
            scope_vid = vid if q["scope"] == "video" else None
            order = np.argsort(-S[qi])
            cands = []
            for j in order:
                r = recs[j]
                if scope_vid and r["L0"]["video_id"] != scope_vid:
                    continue
                cands.append((j, float(S[qi, j])))
            if typ in ("existence", "negation"):
                ans = {"value": "yes" if (cands and cands[0][1] >= TAU) else "no"}
            elif typ == "temporal":
                if not cands:
                    ans = {"spans": []}
                else:
                    r = recs[cands[0][0]]
                    eid = r["L0"]["event_id"]
                    spans = (refined.get(eid, {}).get("inner0.8") or [])
                    spans += (refined.get(eid, {}).get("win15") or [])
                    spans += [[r["L0"]["start_sec"], r["L0"]["end_sec"]]]
                    ans = {"spans": spans}
            elif typ == "retrieval":
                seen, ranked = set(), []
                for j, sc in cands:
                    v = recs[j]["L0"]["video_id"]
                    if v not in seen:
                        seen.add(v)
                        ranked.append(v)
                    if len(ranked) >= 50:
                        break
                ans = {"ranked_videos": ranked}
            else:
                continue
            results.append({"query_id": q["query_id"], "type": typ,
                            "dataset": q["dataset"], "pred": ans})
        m = evaluate(queries, results)["per_type"]
        return (m["existence_plus_negation"]["balanced_acc"],
                m["retrieval"]["recall_at_10"])

    base_bal, base_r10 = run_eval()
    print(f"baseline bal={base_bal} R@10={base_r10}", flush=True)

    # retrieval frequency f_j: top-3 per query (any scope)
    S0 = score_matrix()
    f = np.zeros(n_ev)
    for qi, q in enumerate(queries):
        vid = q["relevant_videos"][0] if q["relevant_videos"] else None
        scope_vid = vid if q["scope"] == "video" else None
        cnt = 0
        for j in np.argsort(-S0[qi]):
            if scope_vid and recs[j]["L0"]["video_id"] != scope_vid:
                continue
            f[j] += 1
            cnt += 1
            if cnt >= TOP_R:
                break

    # marginal value v_j: full LOO balanced-acc drop (749 x eval, numpy only)
    v = np.zeros(n_ev)
    for j in range(n_ev):
        if f[j] == 0:
            v[j] = 0.0  # never retrieved -> zero value on this query set
            continue
        bal_j, _ = run_eval(mask=j)
        v[j] = base_bal - bal_j
        if (j + 1) % 100 == 0:
            print(f"LOO {j+1}/{n_ev}", flush=True)

    # bytes per record
    adir = os.path.join(INGEST, "artifacts")
    line_bytes = {}
    for p in glob.glob(os.path.join(adir, "memory_*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            line_bytes[r["L0"]["event_id"]] = len(l.encode())
    bytes_j = np.array([line_bytes.get(recs[j]["L0"]["event_id"], 1200)
                        + EMB_BYTES for j in range(n_ev)])
    vpb = v / bytes_j

    out = {
        "n_events": n_ev, "n_queries": n_q, "baseline_bal_acc": base_bal,
        "baseline_r10": base_r10, "top_r": TOP_R, "metric_for_v": "balanced_acc",
        "bytes_model": "memory JSONL line bytes + 1536B fp32 embedding",
        "f": f.tolist(), "v": v.tolist(), "bytes": bytes_j.tolist(),
        "v_per_byte": vpb.tolist(),
        "gini_f": gini(f), "gini_v": gini(v), "gini_vpb": gini(vpb),
        "gini_v_nonzero": gini(v[v > 0]),
        "frac_never_retrieved": float(np.mean(f == 0)),
        "frac_zero_value": float(np.mean(v == 0)),
    }

    # uniform vs oracle water-filling (uniform bytes -> random vs value-ranked)
    budgets = [0.9, 0.75, 0.5, 0.25, 0.1]
    uni, ora = {}, {}
    order = np.argsort(-vpb)
    for b in budgets:
        k = max(1, int(round(n_ev * b)))
        oracle_keep = order[:k]
        bals, r10s = [], []
        rng = np.random.RandomState(0)
        for sd in range(3):
            keep = rng.choice(n_ev, size=k, replace=False)
            mask = np.setdiff1d(np.arange(n_ev), keep)
            bal, r10 = run_eval(mask=mask)
            bals.append(bal)
            r10s.append(r10)
        mask_o = np.setdiff1d(np.arange(n_ev), oracle_keep)
        bal_o, r10_o = run_eval(mask=mask_o)
        uni[b] = {"bal": float(np.mean(bals)), "bal_std": float(np.std(bals)),
                  "r10": float(np.mean(r10s))}
        ora[b] = {"bal": bal_o, "r10": r10_o}
        print(f"budget {b}: uniform bal={np.mean(bals):.3f} "
              f"oracle bal={bal_o:.3f} (R@10 {np.mean(r10s):.3f} vs {r10_o:.3f})",
              flush=True)
    out["uniform_vs_oracle"] = {"budgets": budgets, "uniform": uni,
                                "oracle": ora,
                                "uniform_metric": "mean of 3 random subsets (uniform bytes)"}

    with open(os.path.join(args.run_dir, "artifacts", "t1_smb_values.json"),
              "w") as fo:
        json.dump(out, fo)
    print("DONE", json.dumps({k: out[k] for k in
                              ("gini_f", "gini_v", "gini_vpb",
                               "frac_never_retrieved")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
