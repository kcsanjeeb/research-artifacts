# E3 data manifest (updated as downloads complete; sha256 added at completion)

| Dataset | Repo/source | Files pulled | Bytes | Location | sha256 | Status |
|---|---|---|---|---|---|---|
| StreamingBench code+QA | codeload.github.com/THUNLP-MT/StreamingBench @main (tarball 2026-09-27) | full repo | ~26 MB (tarball) | /data2/zhuotaotian2_e2_data/streamingbench/ | (pending) | DONE |
| OVO-Bench src videos | hf-mirror JoeLeelyf/OVO-Bench | README.md + src_videos.tar.partaa..ae (46.3 GB) | 46.3 GB | /data1/zhuotaotian2_e2_data/ovo_bench/ | (pending) | DOWNLOADING |
| OVO-Bench chunked videos | same | chunked_videos.tar.part* (156 GB) | 156 GB | — | — | DEFERRED (disk) |
| MLVU test | hf-mirror MLVU/MLVU_Test | full snapshot | 75.3 GB | /data2/zhuotaotian2_e2_data/mlvu/ | (pending) | QUEUED |
| Video-MME | hf-mirror lmms-lab/Video-MME | README + subtitle.zip + videomme/*.parquet now; videos_chunked_*.zip gated on disk | ~0.01 GB now / 101 GB full | /data2/zhuotaotian2_e2_data/videomme/ | (pending) | PARTIAL |

Downloaders: code/scripts_e3/download_datasets.sh (phase 1, pid 1494856),
download_datasets_p2.sh (phase 2, pid 1505006). Status files in
runs/20260927_0000_e3_phase0/logs/datasets{,_p2}.status.
HF endpoint: https://hf-mirror.com, HF_HUB_DISABLE_XET=1.
