''' . '''
from pathlib import Path

'''Split Dataset for testing'''
from sklearn.model_selection import train_test_split

'''For transforming "PULocationID", "DOLocationID" without chaning the trip distance (just parsing)'''
from sklearn.compose import ColumnTransformer

'''For tansforming "PULocationID", "DOLocationID" to binary values'''
from sklearn.preprocessing import OneHotEncoder

'''Pipeline'''
from sklearn.pipeline import Pipeline

''' and model class'''
from sklearn.ensemble import RandomForestRegressor

'''For calculation RSME'''
from sklearn.metrics import root_mean_squared_error

'''Administring and documenting results of trained models'''
import mlflow

'''From other script files'''
from src.data import load_data, clean_data


def main() -> None:
    download_dir = Path("notebooks/data/raw/")
    download_name = download_dir / "yellow_tripdata_2025_01.parquet"

    df = load_data(download_name)
    df_clean = clean_data(df)

    print("Rows after cleaning:", len(df_clean))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ny_taxi_duration")

    train(df_clean)


def train(df):
    print("--- Generating training and vaildation data set ---")
    features = ["PULocationID", "DOLocationID", "trip_distance"]
    X = df[features].copy()
    y = df["duration"].copy()

    X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

    X_train_small = X_train.sample(n=50_000, random_state=42)
    y_train_small = y_train.loc[X_train_small.index]

    categorical_features = ["PULocationID", "DOLocationID"]
    preprocessor = ColumnTransformer(
        transformers = [
            ("locations", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough"
    )

    print("--- Creating pipeline ---")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(max_depth=20, min_samples_leaf=5, random_state=42))
    ])

    ''' Start mlflow'''
    with mlflow.start_run(run_name="random-forest-script"):
        print("--- Training model ---")
        pipeline.fit(X_train_small,y_train_small)

        print("--- Model starts predicting on data ---")
        y_pred = pipeline.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        print(f"Done, RMSE: {rmse} from {len(X_val)} rows")

        mlflow.log_params({"training_rows": len(X_train_small), "max_depth": 20, "min_samples_leaf": 5, "random_state": 42})
        mlflow.log_metric("rmse_validation", rmse)
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="ny_taxi_pipeline",
            input_example=X_train_small.head(5),
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        print("Saved model URI:", model_info.model_uri)


if __name__=="__main__":
    main()