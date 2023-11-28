from pipelines.deployment_pipeline import continuous_deployment_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService

import click

from typing import cast

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "--c",
    type=click.Choice([DEPLOY,PREDICT,DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Optionally you can choose to only run the deployment"
    "pipeline to train and deploy a model (`deploy`), or to "
    "(`predict`). By default both will be run "
    "(`deploy_and_predict`).",


)
@click.option(

    "--min-accuracy",
    default=0.92,
    help="Minimum accuracy required to deploy the model",
)

def run_deployment(config: str, min_accuracy: float):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy=config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict=config == PREDICT or config == DEPLOY_AND_PREDICT
    if deploy:
        continuous_deployment_pipeline(
            data_path = "D:\Machine Learning Projects\Zen_ML\data\olist_customers_dataset.csv",
            min_accuracy = min_accuracy,
            workers= 3,
            timeout=60,
            )
    # if predict:
        # inference_pipeline()
    
    # print the instructions from zenml 

    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                "The model is running"
            )
        elif service.is_failed:
            print(
                "The MLflow prediction server is in failed state: \n."
            )
        else:
            
            print(
                "No MLflow prediction server is currently running."
            )

if __name__=="__main__":
    run_deployment()