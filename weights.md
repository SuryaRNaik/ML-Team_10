# PathSnap — Model Weights and Training Configuration

This file documents the model configuration and numerical parameters used by PathSnap. It also explains where the **learned model weights** are stored and how to export them.

> **Important distinction:** PathSnap's final classifier is a Random Forest. Unlike a linear model, it does not have one small coefficient/weight for each input feature. Its learned parameters are the structures and values inside its decision trees. The trained classifier is stored in `models/RandomForest.joblib`.

## 1. Final trained model

The final training summary identifies **RandomForest** as the selected model.

| Parameter | Value |
|---|---:|
| Number of trees (`n_estimators`) | 100 |
| Maximum tree depth (`max_depth`) | 16 |
| Minimum samples per leaf (`min_samples_leaf`) | 5 |
| Random seed (`random_state`) | 42 |
| Training trips | 80 |
| Observations per trip | 300 |
| Total training observations | 24,000 |
| Candidate rows | 362,522 |
| Grouped cross-validation folds | 5 |

The model artifact is:

```text
models/RandomForest.joblib
```

## 2. Input features

The classifier uses the following 19 features:

1. `dist`
2. `dist_rel`
3. `dist_prev`
4. `dist_next`
5. `dist_ctx_mean`
6. `t_frac`
7. `along_m`
8. `dist_to_end_m`
9. `arc_len`
10. `cos_ang`
11. `abs_cos_ang`
12. `rank`
13. `n_cands`
14. `speed`
15. `heading_valid`
16. `gap_flag`
17. `out_deg_to`
18. `in_deg_from`
19. `has_reverse`

These are input feature names, not learned weights. The Random Forest learns how to split on these features while fitting its trees.

## 3. HMM parameters

The Hidden Markov Model (HMM) / sequence-decoding configuration is:

| Parameter | Value |
|---|---:|
| `alpha` | 0.5798332898894308 |
| `beta_m` | 5.900917065019388 |
| `log_disc` | -15.312523560078986 |
| `uturn_pen` | 9.92375564055837 |
| `gap_break_s` | 60.0 |
| `p_floor` | 0.000001 |
| `max_hops` | 6 |
| `max_route_m` | 600.0 |

These values are defined in:

```text
configs/hmm.json
```

## 4. Preprocessing and candidate-generation parameters

| Parameter | Value |
|---|---:|
| `candidate_radius_m` | 100.0 |
| `max_candidates_per_point` | 20 |
| `heading_half_window` | 3 |
| `context_offset` | 5 |
| `gap_threshold_s` | 5.0 |
| `min_move_for_heading_m` | 2.0 |
| `label_max_dist_m` | 50.0 |
| `align_dist_cap_m` | 200.0 |
| `align_max_skip` | 3 |
| `align_skip_penalty_m` | 20.0 |

These values are defined in:

```text
configs/preprocess.json
```

## 5. Reported evaluation results

The final training summary reports the following results on the `student_test` partition (19,131 usable observations):

| Metric | Random Forest classifier only | Random Forest + HMM |
|---|---:|---:|
| Top-1 accuracy | 0.8736082797553708 | 0.942135800533166 |
| Continuity | 0.9284967046762214 | 0.9825818600271995 |
| Route F1 | 0.878359944349287 | 0.9743066468082642 |
| Route precision | 0.8017850628030457 | 0.9551629056458836 |
| Route recall | 0.9772988294757188 | 0.9945977249711904 |

These are evaluation metrics, not model weights.

## 6. Where the actual learned Random Forest weights are stored

The fitted tree parameters are serialized in:

```text
models/RandomForest.joblib
```

For a Random Forest, the learned parameters include (among other things):

- the feature selected at each decision-tree split;
- the threshold used at each split;
- the left and right child-node relationships;
- node impurity and sample counts;
- class counts / class distributions at leaf nodes.

The model's `feature_importances_` values can also be exported. These provide a global measure of how much each feature contributes to the forest's split decisions; they are not the complete set of learned tree parameters.

### Exporting the learned parameters

The repository includes the helper script:

```text
export_pathsnap_forest_weights.py
```

Run it from the repository root:

```bash
python export_pathsnap_forest_weights.py
```

It exports:

```text
PathSnap_RF_feature_importances.csv
PathSnap_RF_all_tree_nodes.csv
```

The first CSV contains feature importances. The second contains the node-level information for all trees, including split-feature indices, thresholds, child-node indices, impurity, sample counts, and class counts.

## 7. Other model configurations used for comparison

The experiment script also defines these comparison models. These are experiment configurations; the final training summary names RandomForest as the final model.

| Model | Configuration |
|---|---|
| Logistic Regression | `StandardScaler` + `LogisticRegression(C=1.0, class_weight="balanced", max_iter=300)` |
| Gaussian Naive Bayes | `StandardScaler` + `GaussianNB(var_smoothing=1e-9)` |
| Decision Tree | `max_depth=12`, `min_samples_leaf=20`, `random_state=42` |
| Random Forest | `n_estimators=100`, `max_depth=16`, `min_samples_leaf=5`, `n_jobs=-1`, `random_state=42` |
| Extra Trees | `n_estimators=100`, `max_depth=18`, `min_samples_leaf=5`, `n_jobs=-1`, `random_state=42` |
| Histogram Gradient Boosting | `learning_rate=0.1`, `max_iter=200`, `max_leaf_nodes=31`, `random_state=42` |
| MLP | `hidden_layer_sizes=(64, 32)`, `early_stopping=True`, `max_iter=60`, `random_state=42`; default ReLU activation and Adam solver |

## 8. Data and reproducibility notes

According to the final training summary:

- Raw and processed datasets are not committed to the repository.
- Test labels were not used for training or tuning.
- The final model uses random seed `42`.
- Processed data is generated by preprocessing and stored as Parquet files (for example, `data/processed/points_*.parquet` and `data/processed/candidates_*.parquet`).

## 9. File-format note

- `.joblib` is the format currently used by the repository for the trained scikit-learn model.
- `.pkl` is another possible Python pickle format, but converting the file to `.pkl` does not reveal or change the learned weights; it only changes the serialization file format.
- `.parquet` is used for tabular processed datasets. `.pd` is not a standard pandas file extension.

---

**Summary:** This document records the numerical training/configuration parameters available from the repository metadata. The complete fitted Random Forest parameters remain inside `models/RandomForest.joblib`; use the exporter script to produce the detailed CSV files from that model artifact.
