#!/bin/bash
# XD test video downloader v2 (wget; resume-safe), run inside fleetvad-vlm env
cd /home/san/FleetVAD/research/e0/data
python - <<'PYEOF'
import subprocess, os, urllib.parse, json, time
names = subprocess.check_output(["curl","-s","-m","30","https://hf-mirror.com/api/datasets/jherng/xd-violence/tree/main/data/video/test_videos"])
files = [x["path"].split("/")[-1] for x in json.loads(names)]
base = "https://hf-mirror.com/datasets/jherng/xd-violence/resolve/main/data/video/test_videos/"
have = [f for f in files if os.path.exists(os.path.join("xd_test_videos",f)) and os.path.getsize(os.path.join("xd_test_videos",f))>100000]
print("already have:", len(have), flush=True)
ok, fail = 0, []
for fn in files:
    out = os.path.join("xd_test_videos", fn)
    if os.path.exists(out) and os.path.getsize(out) > 100000:
        continue
    url = base + urllib.parse.quote(fn)
    good = False
    for attempt in range(4):
        r = subprocess.run(["wget","-q","-T","300","-O",out,url])
        if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 100000:
            good = True
            break
        time.sleep(3)
    if good: ok += 1
    else: fail.append(fn)
    if ok and ok % 25 == 0: print("new:", ok, flush=True)
print("XD_VIDEOS_FINAL new:", ok, "fail:", len(fail), flush=True)
if fail: print(fail[:10], flush=True)
PYEOF
