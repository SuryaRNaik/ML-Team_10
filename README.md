# PathSnap — Distributed Map-Matching & Route Reconstruction

PathSnap reconstructs a realistic vehicle route by matching noisy GPS observations to candidate road segments and decoding a connected sequence with an HMM/Viterbi algorithm.

## Repository layout

- src/pathsnap/data_io.py — readers for .track, .nodes, .arcs and .route files.
- src/pathsnap/geo.py — local projection and point-to-road geometry calculations.
- src/pathsnap/prepare.py — cleaning, candidate generation, route-label alignment and feature engineering.
- src/pathsnap/hmm.py — road graph, HMM lattice, Viterbi decoding and route metrics.
- scripts/01_eda.py — exploratory data analysis and Folium maps.
- scripts/02_preprocess.py — preprocessing for training and student-testing partitions.
- scripts/03_train_compare.py — grouped cross-validation and model comparison.
- scripts/04_hmm_decode.py — classifier-only versus HMM/Viterbi evaluation and decoder tuning.
- scripts/serve_api.py — Laptop 1 FastAPI service for the two-laptop evaluation setup.
- ui/ — Laptop 2 static evaluator UI.

Raw and processed datasets are excluded from version control. Keep the evaluator reference routes on Laptop 1.

## ML pipeline

Install project dependencies:

~~~~powershell
python -m pip install -r requirements.txt
~~~~

Place the training and development partitions in the corresponding folders under data/raw/ (folder names can be overridden in the scripts):

~~~~text
data/raw/
  train/
    <trip-id>/<trip-id>.track, .nodes, .arcs, .route
  student_test/
    <trip-id>/<trip-id>.track, .nodes, .arcs, .route
~~~~

Run preprocessing, EDA and model comparison:

~~~~powershell
python scripts/02_preprocess.py
python scripts/01_eda.py
python scripts/03_train_compare.py
~~~~

The model comparison uses trip-grouped cross-validation on the 80 training trips and development evaluation on the 10 student-testing trips. The 10 evaluator-testing trips must not be used for training, feature selection, model selection or hyperparameter tuning.

The validated end-to-end candidate in the current experiment is RandomForest with the tuned HMM/Viterbi decoder. The reported development result is 94.21% Top-1 accuracy on the 10 student-testing trips; it is not an evaluator-test result. See reports/hmm_comparison.csv and reports/final_training_summary.json for the recorded results.

## Two-laptop deployment

The hackathon guide specifies this division:

- **Laptop 1 — Model and API:** owns the trained model, HMM/Viterbi decoder, road-network data, evaluator trips and server-side evaluation process.
- **Laptop 2 — User Interface:** displays the ten evaluator-trip options, sends only the selected trip ID to Laptop 1, and displays the returned predicted route and scores.

### Laptop 1: start the FastAPI backend

Make sure the evaluator partition is present on Laptop 1. By default, the backend expects it at data/raw/evaluator_test, with each trip in its own folder and all four files present. If the provided dataset uses a different folder name, set the environment variable to the exact directory containing the ten evaluator trip folders.

~~~~powershell
# Use the actual evaluator partition folder provided with the dataset.
$env:PATHSNAP_EVALUATOR_DIR = "data/raw/evaluator_test"
$env:PATHSNAP_MODEL = "RandomForest"
python scripts/serve_api.py
~~~~

The service listens on 0.0.0.0:8000, so another laptop on the same network can reach it. Find Laptop 1's IPv4 address with ipconfig. Allow inbound TCP port 8000 through Windows Firewall on the private network when prompted or when required by your setup.

Verify on Laptop 1:

- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

### Laptop 2: start the evaluator UI

From the repository folder:

~~~~powershell
python -m http.server 8080 --directory ui
~~~~

Open http://localhost:8080. In the **Connection** tab, enter Laptop 1's address, for example http://192.168.1.20:8000, then select **Save and test**. Load the evaluator trips, select one and run HMM prediction.

### API contract

- GET /health — confirms that the configured model and evaluator data directory are available.
- GET /trips — returns evaluator trip IDs and GPS point counts only.
- POST /predict — accepts JSON { "trip_id": "<trip-id>" } and returns GPS points, predicted route geometry, classifier/HMM scores and data-quality metadata.
- GET /metrics — returns the saved development HMM comparison from reports/hmm_comparison.csv.

The prediction pass calls prepare_trip(..., with_labels=False). Only after predictions are created does Laptop 1 separately read the evaluator .route file and calculate server-side metrics. The reference route and true arc labels are never included in the API response or sent to Laptop 2.

## Backend contract tests

Install the test dependencies and run:

~~~~powershell
python -m pip install pytest httpx
python -m pytest -q tests/test_serve_api.py
~~~~

The tests use a synthetic trip and mocked model/decoder functions to check API readiness, evaluator-trip listing, prediction response shape, server-side scoring and path-traversal rejection. They do not replace a live integration test using the actual evaluator dataset on both laptops.
