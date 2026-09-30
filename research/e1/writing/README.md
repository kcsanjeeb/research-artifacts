# FleetMem paper draft — Overleaf instructions

Upload the ENTIRE `writing/` folder to Overleaf (New Project → Upload Project → zip this folder, or drag-and-drop preserving structure). Compile with **pdfLaTeX** (Overleaf default). No other configuration needed.

## Structure

```
main.tex               # document class (acmart sigconf, anonymous), abstract, includes
sections/
  01_introduction.tex  # Ch.1 Introduction
  02_related.tex       # Ch.2 Related Theory
  03_gap_formulation.tex # Ch.3 Gap & Formulation
  04_design.tex        # Ch.4 System Design
  05_experiments.tex   # Ch.5 Experiments (all tables/figures)
  06_discussion.tex    # Ch.6 Discussion
  07_conclusion.tex    # Conclusion
  08_appendix.tex      # Appendices A–F
references.bib         # verified bibliography (authors/venues checked against arXiv/DBLP)
figures/               # write_frontier.png, fleet_v1_curves.png, eviction_frontier.png, generality_frontier.png
```

## Data provenance

Every number in this draft traces to a logged run on the server (`~/e1/runs/<run-id>/` with config/env/code-state/cost) and a report mirrored under `research/e1/`. The mapping:

| Paper element | Source |
|---|---|
| Write-cost frontier (Fig. 1, §5.3) | runs `20260918_004{7,8}_writepath_*`, `research/e1/prototype/` |
| SMB benchmark (§5.2) | `research/e1/smb/` |
| Groundedness (§5.2) | `research/e1/narration_spike/groundedness_report.md` |
| v0→v0.1 (Table 4) | `research/e1/querypath/`, `research/e1/querypath_v01/` |
| Baselines (Table 5) | `research/e1/baselines/BASELINE_REPORT.md` |
| Fleet (Table 6, Fig. 2) | `research/e1/fleet/FLEET_REPORT.md` |
| Eviction (Table 7, Fig. 3) | `research/e1/eviction/EVICTION_REPORT.md` |
| TASTI-lite (Table 8) | `research/e1/generality/TASTI_LITE_REPORT.md` |
| Generality (Table 9) | `research/e1/generality/GENERALITY_REPORT.md` |
| Appendix A (E0 numbers) | `Docs/E0_ANALYSIS.md` |

## Known TODOs before submission

- Abstract/intro headline numbers: update if v1.1 (rank-normalized salience) or SMB v1 (SHT normal videos) change them.
- `\sys` name: FleetMem is a placeholder — rename via the `\sys` macro in main.tex only.
- Venue formatting: currently acmart sigconf anonymous; switch class for the chosen venue.
- Figure quality: current PNGs are matplotlib defaults — regenerate at higher DPI for camera-ready.
