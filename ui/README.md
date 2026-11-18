# PathSnap two-system UI

Laptop 1 owns the raw trips, reference routes, trained model and HMM decoder.
Laptop 2 runs this static UI and never receives `.route` files or ground-truth
arc labels.

## Laptop 1: API and model

Place the normalized data under `data/raw/student_test`, install dependencies,
and run:

```powershell
python scripts\serve_api.py
```

The API listens on `0.0.0.0:8000`. For another data location, set
`PATHSNAP_STUDENT_ROOT` to the directory containing the ten trip folders.
`PATHSNAP_MODEL` defaults to `RandomForest`.

## Laptop 2: evaluator UI

```powershell
python -m http.server 8080 --directory ui
```

Open `http://localhost:8080`, enter Laptop 1's address such as
`http://192.168.1.20:8000`, load the ten trips, and select a trip.

## Contract

- `GET /health` checks the model/API connection.
- `GET /trips` returns trip IDs and point counts only.
- `POST /predict` accepts `{ "trip_id": "..." }`.
- `GET /metrics` returns aggregate experiment metrics.

The prediction response contains GPS points, predicted route geometry and
aggregate classifier/HMM metrics. Reference routes remain server-side.
