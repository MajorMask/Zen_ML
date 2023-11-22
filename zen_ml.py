from zenml import pipeline, step
import some_ml_library

# Step 1: Prepare the script (data)

@step
def data_preparation():
    data = some_ml_library.load_data()
    return data

# Step 2: Casting (Feature Engineering) 

@step
def feature_engineering(data):
    features = some_ml_library.extract_feature(data)
    return features

# Step 3: Filming (Model Training)

@step
def model_training(features):
    model = some_ml_library.train_model(features)

#Step 4: Editing (Model Evaluation)

@step
def model_ebaluation(model, features):
    evaluation = some_ml_library.evaluate_model(model, features)
    return evaluation

#Step 5: Distribution (Model Deployment)

@step
def model_deployment(model):
    some_ml_library.deploy_model(model)

# Define the pipeline

@pipeline
def movie_production_pipeline():
    data = data_preparation()
    features = feature_engineering(data)
    model = model_training(features)
    evaluation = model_evaluation(model, features)
    model_deployment(model)

if __name__ == "__main__":
    movie_production_pipeline()
    