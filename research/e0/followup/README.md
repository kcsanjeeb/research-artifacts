# E0 follow-up measurements

Five scripts answering questions E0's design could not, using only artifacts already in
`../results/`. Analysis and conclusions: `Docs/E0_ANALYSIS.md`.

| script | question | key result |
|---|---|---|
| `ucf_xvid.py` | Reproduce E0's `alpha=0` curve, then re-run with same-video candidates excluded | Reproduces exactly (0.474 hit / 0.954 agreement at theta 0.96). Cross-video agreement 0.82-0.87, **below** the 0.922 constant-"normal" baseline |
| `safety.py` | Is `label_agreement >= 0.95` meaningful at a 7.8% base rate? | No. At the passing operating point, 9.9% of all anomalous clips are suppressed; kappa 0.66 |
| `safetylaw.py` | Does the master plan's rise-gate safety law prevent that? | Barely. rho=0 halves the hit rate and still suppresses 4% of anomalies — VadCLIP scores are flat mid-event |
| `marginal.py` | Hit rate over clips that actually cost a VLM call | 13.9% of escalations at <=1.3% suppression, 95% same-video → 1.16x capacity |
| `xcam.py` | ShanghaiTech with same-**camera** candidates excluded | Cross-camera matches abundant (71-83% of clips) but agreement 0.55-0.59 vs a 0.501 chance baseline |

Deterministic, no GPU, ~30 s total. Each writes a log (and where applicable a JSON) to
`results/`.
