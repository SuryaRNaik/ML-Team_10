# PathSnap – preprocessing stage

## Run (≈20 s total)
```bash
pip install -r requirements.txt
# put the raw folders at data/raw/train and data/raw/student_test
python scripts/02_preprocess.py     # -> data/processed/{points,candidates}_<partition>.parquet + trip_summary_*.csv
python scripts/01_eda.py            # -> reports/ (inventory csv, figures, folium maps, data_quality_log.md)
```

## Layout
- `src/pathsnap/data_io.py`  raw readers (.track/.nodes/.arcs/.route)
- `src/pathsnap/geo.py`      per-trip UTM projection, point-to-segment distance
- `src/pathsnap/prepare.py`  cleaning, candidate generation, route alignment (labels), features. `prepare_trip()` is reused by the API at inference time (call with `with_labels=False`)
- `src/pathsnap/viz.py`      Folium map builder (GPS, network, reference route, predicted route)
- `configs/preprocess.json`  radius, K, windows, thresholds
- `reports/`                 inventory, figures, maps, data_quality_log.md

## Output tables
`candidates_*.parquet`: one row per (trip_id, pt_idx, arc_id). Target = `label`. Train/evaluate only on `label_ok == 1`.
Group CV by `trip_id`. Features: dist, dist_rel, dist_prev, dist_next, dist_ctx_mean, t_frac, along_m, dist_to_end_m, arc_len,
cos_ang, abs_cos_ang, rank, n_cands, speed, heading_valid, gap_flag, out_deg_to, in_deg_from, has_reverse.
`points_*.parquet`: one row per GPS observation (x,y metres, kinematics, true_arc, label_ok, true_in_cands) – needed by the HMM/decoder metrics.

## HMM / Viterbi route reconstruction
- `src/pathsnap/hmm.py`  hidden states = candidate arcs per GPS point; emission = alpha*log(classifier prob);
  transition = -|route distance - GPS distance|/beta (connected arcs), constant penalty if disconnected, U-turn penalty;
  chain restarts at GPS gaps; points with no candidates are filled from the nearest decoded arc.
- `configs/hmm.json`    decoder parameters (overwritten by `--tune`)
- `scripts/04_hmm_decode.py`
```bash
python scripts/04_hmm_decode.py --model HistGradientBoosting                 # classifier-only vs +HMM on student_test
python scripts/04_hmm_decode.py --model HistGradientBoosting --tune          # tune decoder on TRAIN (out-of-fold) first
```
Outputs: `reports/hmm_comparison.csv`, `hmm_per_trip.csv`, `hmm_tuning.csv`, `figures/07_hmm_vs_classifier.png`, `maps/hmm_*.html`.
The API calls `build_lattice` / `decode_hmm` from `hmm.py` with `prepare_trip(..., with_labels=False)` output.
