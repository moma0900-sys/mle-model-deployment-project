# NYC Yellow Taxi – Trip Duration Prediction

This project predicts the duration of an NYC Yellow Taxi trip using its pickup zone, drop-off zone, and distance.

The workflow covers data cleaning, model training, experiment tracking with MLflow, and serving predictions through a local FastAPI application.

**Result:** The tuned Random Forest model achieves a validation RMSE of **4.882 minutes**.

## Technologies

- Python
- pandas and PyArrow
- scikit-learn
- MLflow with a local SQLite database
- FastAPI and Pydantic
- Uvicorn
- requests
- uv for Python environment management

## Project Structure

```text
.
├── notebooks/
│   ├── 01_exploring_data.ipynb
│   ├── 02_training_model.ipynb
│   └── data/
│       ├── raw/
│       │   └── yellow_tripdata_2025_01.parquet
│       └── cleaned/
├── src/
│   └── data.py
├── train.py
├── pipeline.py
├── main.py
├── api.py
├── client.py
├── sample-request.json
├── pyproject.toml
├── uv.lock
└── README.md
```

The data directories contain files downloaded or generated locally. MLflow additionally creates a local database and model artifacts.

| File | Purpose |
|---|---|
| `src/data.py` | Load and clean trip data |
| `train.py` | Split data, train and evaluate the model, and save it with MLflow |
| `pipeline.py` | Load a saved pipeline and generate predictions |
| `main.py` | Demonstrate a direct prediction without an HTTP request |
| `api.py` | Provide the FastAPI application and `/predict` endpoint |
| `client.py` | Send a sample request to the API |
| `sample-request.json` | Store the sample request data |

## Dataset and Cleaning

The project uses the **NYC Yellow Taxi Trip Records for January 2025**.

Source: [NYC Taxi & Limousine Commission – Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

The target variable, `duration`, is calculated as the difference between drop-off and pickup timestamps, expressed in minutes:

```python
duration = (
    df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
).dt.total_seconds() / 60
```

Trips are retained if they meet both conditions:

- Duration between **1 and 60 minutes**, inclusive
- Distance between **0.5 and 20 miles**, inclusive

| Processing Stage | Remaining Trips |
|---|---:|
| Raw data | 3,475,226 |
| After filtering trip duration | 3,403,248 |
| After additionally filtering trip distance | 3,174,647 |

The duration filter removes 71,978 trips, or 2.07% of the raw data. The subsequent distance filter removes another 228,601 trips, or 6.72% of the data remaining after the duration filter.

These limits define the scope of the model. Excluded trips are not necessarily incorrect records.

## Model and Training Procedure

The model uses three input features:

| Feature | Description |
|---|---|
| `PULocationID` | Pickup zone ID |
| `DOLocationID` | Drop-off zone ID |
| `trip_distance` | Trip distance in miles |

The two location IDs are processed using `OneHotEncoder(handle_unknown="ignore")`. Trip distance is passed through unchanged.

Preprocessing and the `RandomForestRegressor` are combined in a single scikit-learn pipeline. This ensures that the same preprocessing is applied during training and prediction.

The training procedure is:

1. Split the cleaned data into 80% training data and 20% validation data.
2. Sample 50,000 trips from the training data.
3. Select the corresponding target values using the same row indices.
4. Fit the pipeline on this training sample.
5. Evaluate predictions on the complete validation partition.

The split, training sample, and Random Forest all use `random_state=42`.

### Hyperparameter Tuning

The training notebook uses `GridSearchCV` to evaluate nine parameter combinations with three-fold cross-validation:

```python
max_depth = [10, 20, None]
min_samples_leaf = [1, 5, 20]
```

The best configuration among those evaluated was:

```python
RandomForestRegressor(
    max_depth=20,
    min_samples_leaf=5,
    random_state=42,
)
```

The training script uses these parameters directly. It does not repeat the parameter search.

## Results

| Model | Validation RMSE in Minutes |
|---|---:|
| Constant prediction using the training target mean | 9.422 |
| Random Forest before tuning | 5.062 |
| Tuned Random Forest | **4.882** |

The latest training script run produced an RMSE of `4.881916803570638`.

Tuning reduced validation RMSE by approximately **3.6%** compared with the original Random Forest. The best mean cross-validation RMSE was approximately **4.944 minutes**.

RMSE gives greater weight to large prediction errors. It is neither an upper error bound for individual trips nor the mean absolute error.

The parameter search required additional computation but produced a measurable improvement without changing how the final pipeline is used.

## Running the Project Locally

### 1. Requirements and Installation

Requirements:

- Python 3.13 or newer, as specified in `pyproject.toml`
- uv installed
- A local copy of this repository

Run all commands below in **Windows PowerShell from the project root directory**. This matters because the data paths and SQLite connection are relative to the current working directory.

Install dependencies using the lock file:

```powershell
uv sync --locked
```

This creates the `.venv` environment. The following commands call its Python executable directly, so activating the environment separately is unnecessary.

### 2. Download the Raw Data

On the TLC data page, select **2025 → January → Yellow Taxi Trip Records**.

Alternatively, download the file using PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path "notebooks/data/raw"

Invoke-WebRequest `
    -Uri "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet" `
    -OutFile "notebooks/data/raw/yellow_tripdata_2025_01.parquet"
```

The local filename contains `2025_01` and must match the filename used by the training script.

### 3. Train and Save the Model

```powershell
.\.venv\Scripts\python.exe train.py
```

The script loads and cleans the raw data, trains the pipeline, and calculates validation RMSE.

MLflow is configured as follows:

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ny_taxi_duration")
```

The run named `random-forest-script` records:

- Number of training rows
- `max_depth`
- `min_samples_leaf`
- `random_state`
- Validation RMSE as `rmse_validation`
- The complete pipeline as `ny_taxi_pipeline`

At the end, output similar to the following is displayed:

```text
Done, RMSE: 4.881916803570638
Saved model URI: models:/m-...
```

Each newly logged model receives a new model URI. This URI references a model in the local MLflow environment.

The notebooks do not need to be executed before running the training script.

### 4. Configure the Model URI in the API

Copy the printed model URI into the `model_uri` variable in `api.py`:

```python
model_uri = "models:/REPLACE_WITH_THE_NEW_MODEL_ID"
```

Replace the entire placeholder value with the actual URI printed by the training script.

The API loads the pipeline when the application module is imported, once per server process. Restarting or reloading the application loads it again.

A model URI from another machine only works if the corresponding MLflow database and model artifacts are available. For a fresh local setup, train the model first and use the newly generated URI.

### 5. Start the API

In one terminal, run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --reload
```

Keep this terminal running.

The interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Running `python api.py` alone does not start a web server in the current implementation. Uvicorn starts and serves the application.

### 6. Send a Sample Request

The `sample-request.json` file contains:

```json
{
  "PULocationID": 229,
  "DOLocationID": 237,
  "trip_distance": 1.6
}
```

In a second terminal, also from the project root, run:

```powershell
.\.venv\Scripts\python.exe client.py
```

The client reads the JSON file and sends its contents to:

```text
POST http://127.0.0.1:8000/predict
```

The trained model returned the following successful response:

```json
{
  "predicted_duration_minutes": 10.774348806856695
}
```

The HTTP status is `200`. The predicted duration is approximately **10.77 minutes**.

### 7. Open the MLflow UI

Optionally, start the tracking interface in another terminal:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to inspect the runs, parameters, metrics, and model artifacts in the `ny_taxi_duration` experiment.

The MLflow UI does not need to be running for training or prediction.

## API Input Validation

The API accepts one trip per request. Pydantic validates the required fields:

| Field | Requirement |
|---|---|
| `PULocationID` | Positive integer |
| `DOLocationID` | Positive integer |
| `trip_distance` | Number between 0.5 and 20, inclusive |

Missing required fields or values outside these limits produce HTTP status `422`.

Location IDs are currently checked only for positive integer values. Validation against the official taxi zone directory has not yet been implemented.

## Verification Performed

The following workflows have been checked manually:

- Data cleaning produces 3,174,647 rows.
- The training script reproduces the notebook RMSE of approximately 4.882 minutes.
- A saved pipeline can be loaded from MLflow.
- Predictions before and after saving match for the checked sample inputs.
- The API returns status `200` and a prediction for a valid request.
- An invalid negative distance is rejected with status `422`.
- The Python client successfully communicates with the local API.

These checks do not replace an automated test suite.

## Limitations and Future Improvements

The model was trained using data from one month and a sample of 50,000 training trips. Its reported performance applies to the specified data scope. Generalization to other months or changing traffic conditions has not yet been evaluated.

Actual trip distance is normally known only after a trip has ended. Predictions made before departure would therefore require an estimated route distance. Evaluation using recorded distances only partially represents that use case.

Additional limitations:

- Time of day, day of week, weather, and current traffic conditions are not included.
- The validation data was used for multiple model comparisons. No additional untouched test set has been evaluated.
- Model URIs and local paths are currently hard-coded.
- The application runs locally using Uvicorn in development mode.

With more time, I would:

1. Evaluate the model on a temporally separate test set from a later period.
2. Add features such as pickup hour and day of week.
3. Compare larger training samples in terms of model quality and computation time.
4. Add automated tests for data filtering, model loading, and the API.
5. Manage model URIs and paths through centralized configuration.
6. Analyze prediction errors by trip distance and taxi zone.
7. Add an optional Docker setup for a consistent runtime environment.