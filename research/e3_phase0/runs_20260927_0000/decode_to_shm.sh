#!/bin/bash
RUN=/data3/zhuotaotian2_e2/runs/20260927_0000_e3_phase0
echo "DECODE_START $(date)" > $RUN/decode.status
/data4/rekv/bin/python - <<PY
import json
d = json.load(open("/data3/zhuotaotian2_e2/code/ReKV/data/rvs/ego/ego4d_oe.json"))
import subprocess, sys
procs = []
for v in d:
    vid = v["video_id"]; src = "/data3/zhuotaotian2_e2/code/MuKV/" + v["video_path"]
    procs.append((vid, subprocess.Popen(["/data4/rekv/bin/python", "/data3/zhuotaotian2_e2/runs/20260927_0000_e3_phase0/decode_to_shm.py", vid, src], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
# run max 5 at a time
running = [(v,p) for v,p in procs[:5]]; pending = procs[5:]
while running:
    import time
    time.sleep(10)
    still = []
    for v,p in running:
        if p.poll() is None: still.append((v,p))
        else:
            out = p.stdout.read()
            open(f"/data3/zhuotaotian2_e2/runs/20260927_0000_e3_phase0/logs/decode_{v}.log","w").write(out)
            print(v, "->", out.strip().splitlines()[-1] if out.strip() else "NO OUTPUT", flush=True)
    running = still
    while pending and len(running) < 5:
        v,p = pending.pop(0); running.append((v,p))
print("ALL_DECODE_SUBPROCS_REAPED", flush=True)
PY
echo "DECODE_DONE $(date)" >> $RUN/decode.status
