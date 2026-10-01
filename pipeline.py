'''Administring and documenting results of trained models'''
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

'''Loading tuned model'''
def load_pipeline(model_uri):
    loaded_tuned_pipeline = mlflow.sklearn.load_model(model_uri)
    return loaded_tuned_pipeline


'''Predicting duration on ride_data'''
def predict_duration(pipeline, ride_data):
    predictions = pipeline.predict(ride_data)
    return predictions