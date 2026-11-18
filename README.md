# PathSnap — GPS-to-Road Map Matching with Machine Learning and HMM/Viterbi

PathSnap reconstructs a likely road route from noisy GPS observations. It generates nearby road-arc candidates for each GPS point, scores those candidates with a trained machine-learning classifier, and applies Hidden Markov Model (HMM) decoding with the Viterbi algorithm to produce a route that is more spatially and topologically consistent.

The repository contains the data-preparation pipeline, model-comparison experiments, HMM/Viterbi decoder, evaluation reports, saved models, and a browser-based evaluator that connects to a separate API machine.

## Project workflow

1. **Exploratory data analysis:** inspect raw trip files and produce a dataset inventory, quality logs, figures, and maps.
2. **Preprocessing:** clean GPS observations, project coordinates, generate candidate road arcs, align reference routes for labels, and build model features.
3. **Model training and comparison:** train multiple candidate-scoring models using trip-grouped cross-validation.
4. **Route decoding:** use candidate probabilities as HMM emissions and road-network connectivity/GPS movement as transition information; decode the sequence with Viterbi.
5. **Evaluation and visualisation:** compare classifier-only output with HMM/Viterbi output, generate per-trip metrics and route maps.
6. **Two-system demonstration:** run the model/API on one laptop and the static evaluation UI on another.

## Repository structure

```text
ML-Team_10/
├── configs/
│   ├── hmm.json                 # HMM/Viterbi decoder parameters
│   └── preprocess.json          # Candidate generation and preprocessing settings
├── models/                      # Saved scikit-learn model files (.joblib)
├── reports/
│   ├── figures/                 # Experiment and evaluation plots
│   ├── maps/                    # Generated route maps
│   ├── data_quality_log.md
│   ├── dataset_inventory.csv
│   ├── experiment_log.json
│   ├── model_comparison.csv
│   ├── cv_fold_scores.csv
│   ├── hmm_comparison.csv
│   ├── hmm_per_trip.csv
│   └── final_training_summary.json
├── scripts/
│   ├── 01_eda.py                # Dataset inspection and EDA
│   ├── 02_preprocess.py         # Cleaning, candidate generation and features
│   ├── 03_train_compare.py     # Train and compare candidate-scoring models
│   ├── 04_hmm_decode.py        # HMM/Viterbi decoding and evaluation
│   ├── 05_error_analysis.py    # Analyse remaining route errors
│   ├── 06_bayes_opt.py         # Bayesian hyperparameter optimisation
│   └── serve_api.py            # FastAPI service for the evaluator UI
├── src/
│   └── pathsnap/
│       ├── data_io.py           # Raw trip file readers
│       ├── features2.py         # Additional/group-level features
│       ├── geo.py               # Geographic projection and distance utilities
│       ├── hmm.py               # Road graph, HMM and Viterbi implementation
│       ├── prepare.py           # Trip preparation and candidate features
│       └── viz.py               # Folium route-map visualisation
├── ui/
│   ├── index.html               # Evaluator interface
│   ├── app.js                   # UI behaviour
│   ├── api.js                   # API communication
│   ├── config.js                # UI configuration
│   ├── data.js                  # UI data helpers
│   ├── style.css                # UI styling
│   └── README.md                # Two-system UI instructions
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or later is recommended.
- pip
- Raw trip data in the expected directory/file layout.
- For the two-system evaluator, two computers on a network that can reach the API host.

Create and activate a virtual environment from the repository root.

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux**
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data placement

The pipeline expects separate partitions for training and student-test data under `data/raw/`. Each trip is stored in its own directory and has four files sharing the trip identifier:

```text
data/
└── raw/
    ├── train/
    │   └── <trip_id>/
    │       ├── <trip_id>.track
    │       ├── <trip_id>.nodes
    │       ├── <trip_id>.arcs
    │       └── <trip_id>.route
    └── student_test/
        └── <trip_id>/
            ├── <trip_id>.track
            ├── <trip_id>.nodes
            ├── <trip_id>.arcs
            └── <trip_id>.route
```

The raw dataset is not included in this repository. Obtain it from the authorised project/course source and preserve the expected folder and file naming conventions. The evaluator-test partition must not be used for training or tuning.

## Run the pipeline

Run commands from the repository root.

### 1. Explore the dataset

```bash
python scripts/01_eda.py
```

The script creates dataset inventory and data-quality outputs, along with exploratory figures and maps under `reports/`.

### 2. Preprocess trips

```bash
python scripts/02_preprocess.py
```

Preprocessing creates Parquet point and candidate tables under `data/processed/`. The candidate table contains one row per candidate road arc and GPS observation. The pipeline also produces trip summaries and the features used by the training and decoding stages.

Important configuration values are in `configs/preprocess.json`, including candidate search radius, maximum candidates per point, heading context, gap threshold, and label-alignment settings.

### 3. Train and compare candidate-scoring models

```bash
python scripts/03_train_compare.py
```

The training script compares candidate-scoring classifiers, including Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting, and MLP. It uses GroupKFold split by `trip_id` so observations from the same trip are kept in one fold.

Optional arguments:

```bash
python scripts/03_train_compare.py --obs-per-trip 300 --folds 5
python scripts/03_train_compare.py --only RandomForest HistGradientBoosting
```

The script saves trained models under `models/` and comparison results, fold scores, experiment metadata, and figures under `reports/`.

### 4. Decode routes with HMM/Viterbi

Evaluate classifier-only predictions against HMM/Viterbi-decoded routes on the student-test partition:

```bash
python scripts/04_hmm_decode.py --model RandomForest
```

Tune decoder parameters using training trips only, then evaluate:

```bash
python scripts/04_hmm_decode.py --model RandomForest --tune --tune-trials 60
```

You can also evaluate a model saved at a specific path:

```bash
python scripts/04_hmm_decode.py --model-path models/HistGradientBoosting_tuned.joblib
```

Optional arguments include `--partition train` or `--partition student_test`, and `--map-trips` to choose specific trips for map generation. The default evaluation partition is `student_test`.

The decoder uses:
- **Emission scores:** candidate probabilities from the classifier.
- **Transition scores:** compatibility between road-network route distance and observed GPS movement, with penalties for disconnected transitions and U-turns.
- **Viterbi decoding:** selects the most likely sequence of candidate road arcs over the trip.

Decoder settings are stored in `configs/hmm.json`.

### 5. Error analysis and Bayesian optimisation

Inspect remaining prediction errors:

```bash
python scripts/05_error_analysis.py
```

Run Bayesian optimisation for the HistGradientBoosting model:

```bash
python scripts/06_bayes_opt.py --trials 40 --obs-per-trip 400
```

Then evaluate the tuned model and tune the decoder on training data:

```bash
python scripts/04_hmm_decode.py --model-path models/HistGradientBoosting_tuned.joblib --tune
python scripts/05_error_analysis.py --model-path models/HistGradientBoosting_tuned.joblib
```

Use the exact options supported by each script's `--help` output if your local version differs.

## Saved experiment results

The checked-in reports record the current experiment results. The previous project README documented the following comparison for Random Forest with and without HMM/Viterbi:

| Metric | Random Forest | Random Forest + HMM |
|---|---:|---:|
| Top-1 accuracy | 87.36% | **94.21%** |
| Route continuity | 92.85% | **98.26%** |
| Route F1 | 87.84% | **97.43%** |
| Route precision | 80.18% | **95.52%** |
| Route recall | 97.73% | **99.46%** |

These values are retained from the previous README; consult the checked-in result files for detailed run and per-trip results:

- `reports/hmm_comparison.csv`
- `reports/hmm_per_trip.csv`
- `reports/final_training_summary.json`
- `reports/model_comparison.csv`
- `reports/cv_fold_scores.csv`

## Two-laptop evaluator setup

The project includes a static browser UI and a FastAPI service. The intended split is:

- **Laptop 1 — API/ML host:** keeps the raw student-test trips, reference routes, trained model, road network, and HMM decoder. It runs `scripts/serve_api.py`.
- **Laptop 2 — evaluator UI:** runs the static files in `ui/` and sends a selected trip ID to Laptop 1. Reference route files are not sent to the browser UI.

### Laptop 1: start the API

Ensure the model file, configuration, dependencies, and student-test trip data are present. From the repository root:

```bash
python scripts/serve_api.py
```

By default, the API listens on `0.0.0.0:8000`.

Optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `PATHSNAP_MODEL` | `RandomForest` | Model filename stem in `models/` |
| `PATHSNAP_RAW_ROOT` | `data/raw` | Root containing raw partitions |
| `PATHSNAP_STUDENT_ROOT` | `data/raw/student_test` | Directory containing student-test trip folders |
| `PATHSNAP_API_HOST` | `0.0.0.0` | API bind host |
| `PATHSNAP_API_PORT` | `8000` | API port |

Check the API from Laptop 1:

```text
http://localhost:8000/health
```

Find Laptop 1's LAN IPv4 address and make sure the firewall permits inbound connections on the configured API port.

### Laptop 2: start the UI

From the repository root:

```bash
python -m http.server 8080 --directory ui
```

Open:

```text
http://localhost:8080
```

In the UI's Connection tab, enter Laptop 1's reachable address, for example:

```text
http://192.168.1.20:8000
```

Replace the example IP with Laptop 1's actual LAN address. The two laptops must be able to communicate over the same network or another network that permits the connection.

### API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check API status and loaded model |
| `GET /trips` | Return available trip IDs and point counts |
| `POST /predict` | Run prediction for the selected trip ID |
| `GET /metrics` | Return aggregate experiment metrics |

Example prediction request:

```json
{
  "trip_id": "YOUR_TRIP_ID"
}
```

The API validates the trip ID and reads the corresponding files on the server. Its response includes GPS points, predicted route geometry, candidate predictions, and aggregate classifier-only versus HMM/Viterbi metrics. Reference route files remain on Laptop 1.

For additional UI-specific instructions, see [ui/README.md](ui/README.md).

## Outputs

Depending on the scripts run and available input data, the project writes:

- `data/processed/points_*.parquet` — GPS point-level data and labels.
- `data/processed/candidates_*.parquet` — candidate road arcs and model features.
- `reports/model_comparison.csv` and `reports/cv_fold_scores.csv` — classifier comparison and grouped CV scores.
- `reports/hmm_comparison.csv` and `reports/hmm_per_trip.csv` — classifier-only versus HMM/Viterbi evaluation.
- `reports/hmm_predictions_*.csv` — per-observation decoded predictions.
- `reports/figures/` — experiment/evaluation figures.
- `reports/maps/` — interactive Folium HTML maps.
- `models/*.joblib` — trained scikit-learn models.

Generated datasets and raw trip files are expected to be supplied locally; do not commit private or restricted data unless the project owner explicitly permits it.

## Reproducibility and data-use notes

- Run scripts from the repository root so relative paths resolve as expected.
- Keep raw trip IDs and file names unchanged.
- Use trip-level splits for validation to avoid leakage between observations from the same trip.
- Use the training partition for model fitting and decoder tuning; keep evaluation partitions separate according to the project protocol.
- Keep model, preprocessing, and HMM configuration files together when sharing a runnable experiment.

## Team and project

**Repository:** [SuryaRNaik/ML-Team_10](https://github.com/SuryaRNaik/ML-Team_10)

This README describes the code and commands currently present in the repository. Dataset access, exact environment setup, and experiment outputs depend on the project files supplied to the team.
