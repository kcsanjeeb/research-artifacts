"""T2.4 — honest metric reporting. No GPU, no new experiments.

(a) Existence accuracy ALONE per eviction budget/policy, from the existing C3
    run (~/e1/runs/20260919_1019_eviction_c3) + the T2.3 matrix run
    (~/e1/runs/20260925_1924_t23_tombstone_matrix) — same numbers, tombstones
    do not affect existence.
(b) Cross-camera (SHT) query status: check whether the 30 SHT cross_camera
    SMB queries were ever evaluated; verify whether any SHT memory/tier-1
    scores exist.
"""
import glob
import json
import os
import re
import sys
import time

FLEETMEM = os.path.expanduser("~/e1/fleetmem")
sys.path.insert(0, FLEETMEM)
from make_run_dir import make_run_dir  # noqa: E402

E1 = os.path.expanduser("~/e1")
C3 = os.path.join(E1, "runs/20260919_1019_eviction_c3")
T23 = os.path.join(E1, "runs/20260925_1924_t23_tombstone_matrix")
SMB = os.path.join(E1, "smb")
INGEST = os.path.join(E1, "runs/20260918_0111_writepath_fullcorpus")
TAGRUN = os.path.join(E1, "runs/20260918_0802_v01_tagging")
E0RES = "/home/san/FleetVAD/research/e0/results"

OUT = os.path.join(E1, "t2_integrity")


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---------------- (a) existence accuracy alone ----------------
    c3 = json.load(open(os.path.join(C3, "eviction_summary.json")))
    t23 = json.load(open(os.path.join(T23, "matrix_summary.json")))
    # sanity: C3 existence numbers == T2.3 tombstones-OFF numbers
    t23_off = {(r["policy"], r["budget"], r["seed"]): r for r in t23
               if not r["tombstones"]}
    t23_on = {(r["policy"], r["budget"], r["seed"]): r for r in t23
              if r["tombstones"]}
    mism = []
    for r in c3:
        if r["policy"] == "none":
            continue
        if r["policy"] == "coverage_tombstone":
            ref = t23_on[("coverage", r["budget"], r["seed"])]
        else:
            ref = t23_off[(r["policy"], r["budget"], r["seed"])]
        if abs(ref["existence"] - r["existence"]) > 1e-9:
            mism.append(r)
    existence_table = [r for r in c3]  # policy, budget, seed, existence, ...

    # ---------------- (b) cross-camera status ----------------
    queries = [json.loads(l) for l in open(os.path.join(SMB, "queries.jsonl"))]
    n_xcam = sum(1 for q in queries
                 if q["dataset"] == "sht" and q["type"] == "cross_camera")
    n_sht = sum(1 for q in queries if q["dataset"] == "sht")

    mem_sht = glob.glob(os.path.join(INGEST, "artifacts", "memory_sht_*"))
    mem_all = glob.glob(os.path.join(INGEST, "artifacts", "memory_*.jsonl"))

    # every smb_results jsonl across all runs: does any contain cross_camera?
    xcam_lines = 0
    result_files = glob.glob(os.path.join(E1, "runs", "*", "query_eval",
                                          "smb_results_*.jsonl"))
    for p in result_files:
        for l in open(p):
            if '"cross_camera"' in l:
                xcam_lines += 1

    import numpy as np
    z = np.load(os.path.join(E0RES, "tier1_scores.npz"), allow_pickle=True)
    t1_vids = [str(v) for v in z["video_ids"]]
    sht_meta = None
    sp = os.path.join(E0RES, "sht_features.npz")
    if os.path.exists(sp):
        zs = np.load(sp, allow_pickle=True)
        sht_meta = {"keys": list(zs.keys()),
                    "n_videos": int(len(zs["meta"]))}

    metrics = {
        "existence_per_budget": existence_table,
        "c3_vs_t23_existence_mismatches": len(mism),
        "cross_camera": {
            "sht_queries_total": n_sht,
            "sht_cross_camera_queries": n_xcam,
            "sht_memory_files_in_fullcorpus_run": len(mem_sht),
            "memory_files_total": len(mem_all),
            "smb_results_files_scanned": len(result_files),
            "cross_camera_lines_in_any_results": xcam_lines,
            "tier1_scores_videos": len(t1_vids),
            "tier1_scores_sht_videos": sum(1 for v in t1_vids
                                           if re.match(r"\d\d_\d", v)),
            "sht_features_npz": sht_meta,
            "evaluated": False,
            "status": "BLOCKED",
            "reason": ("SHT was never ingested: the 749-event full-corpus "
                       "memory contains 0 SHT records (1090 memory files, "
                       "UCF+XD only) and no SHT tier-1 scores exist "
                       "(tier1_scores.npz covers 290 UCF test videos only; "
                       "sht_features.npz holds E0 scene/motion redundancy "
                       "features, not tier-1 CLIP scores). 0 cross_camera "
                       "lines in any smb_results file across all runs."),
        },
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    env = {"python": sys.version.split()[0],
           "conda_env": os.environ.get("CONDA_PREFIX", "unknown"),
           "date": time.strftime("%Y-%m-%d %H:%M %Z"),
           "no_gpu": True}
    with open(os.path.join(OUT, "env.json"), "w") as f:
        json.dump(env, f, indent=2)
    config = {
        "task": "T2.4 honest metric reporting",
        "inputs": {"eviction_run": C3, "t23_run": T23, "smb_queries":
                   os.path.join(SMB, "queries.jsonl"),
                   "ingest_run": INGEST, "tagging_run": TAGRUN,
                   "e0_tier1": os.path.join(E0RES, "tier1_scores.npz")},
        "no_new_inference": True,
    }
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    code_state = {"code_dir": FLEETMEM, "files": {}}
    for fn in os.listdir(FLEETMEM):
        if fn.endswith((".py", ".md")):
            p = os.path.join(FLEETMEM, fn)
            code_state["files"][fn] = {"bytes": os.path.getsize(p),
                                       "mtime": os.path.getmtime(p)}
    with open(os.path.join(OUT, "code_state.json"), "w") as f:
        json.dump(code_state, f, indent=2)
    print(json.dumps(metrics["cross_camera"], indent=2))
    print("existence check mismatches:", len(mism))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
