#!/bin/bash
# 2-video subset anno for the oracle grain-policy runs (clairvoyant bound).
PY=/data4/rekv/bin/python
$PY - <<'PYEOF'
import json
anno = json.load(open('/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB/ego4d_oe_npy.json'))
sub = anno[:2]
json.dump(sub, open('/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB/oracle_subset_anno.json', 'w'))
print('subset videos:', [v['video_id'] for v in sub], 'questions:', sum(len(v['conversations']) for v in sub))
PYEOF
