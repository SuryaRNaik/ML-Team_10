# PathSnap two-laptop evaluator UI

The static UI runs on Laptop 2. It communicates with the FastAPI backend on Laptop 1 through ui/api.js.

## Laptop 1 — backend and data

Laptop 1 must contain the trained model, HMM configuration and the evaluator partition. The backend defaults to data/raw/evaluator_test; override this when the supplied dataset uses a different folder name:

~~~~powershell
$env:PATHSNAP_EVALUATOR_DIR = "data/raw/evaluator_test"
$env:PATHSNAP_MODEL = "RandomForest"
python -m pip install -r requirements.txt
python scripts/serve_api.py
~~~~

The API listens on 0.0.0.0:8000. Find Laptop 1's IPv4 address with ipconfig and allow inbound TCP port 8000 on the private network in Windows Firewall if prompted.

Check http://127.0.0.1:8000/health and http://127.0.0.1:8000/docs on Laptop 1.

## Laptop 2 — UI

~~~~powershell
python -m http.server 8080 --directory ui
~~~~

Open http://localhost:8080, go to the **Connection** tab and enter Laptop 1's LAN URL (for example http://192.168.1.20:8000). Select **Save and test**, load the ten evaluator trips, select one and run HMM prediction.

## API contract

- GET /health
- GET /trips — trip IDs and GPS point counts only.
- POST /predict — JSON body { "trip_id": "<trip-id>" }.
- GET /metrics — development classifier/HMM comparison metrics.

Laptop 2 does not upload or receive .route files. Laptop 1 performs prediction without labels, then reads the reference route separately for server-side scoring. Only the resulting metrics are returned; the route labels themselves remain on Laptop 1.
