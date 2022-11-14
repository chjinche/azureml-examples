from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

from azure.ai.ml import MLClient, Input
from azure.ai.ml.dsl import pipeline
from pathlib import Path

from pathlib import Path
from random import randint
from uuid import uuid4

# mldesigner package contains the command_component which can be used to define component from a python function
from mldesigner import command_component, Input, Output


@command_component()
def gen_silos(
    params: str,
    output: Output(type="uri_folder"),
):
    (Path(output) / "silos").write_text(params)


@command_component()
def train_model(
    silo: str,
    output_model: Output(type="uri_folder"),
):
    lines = [
        f"silo: {silo}",
        f"model output path: {output_model}",
    ]

    for line in lines:
        print(line)

    # Do the train and save the trained model as a file into the output folder.
    # Here only output a dummy data for demo.
    model = str(uuid4())
    (Path(output_model) / "model").write_text(model)


@command_component()
def validate(
    model: Input(type="uri_folder"),
    silo: str,
    valid_data: Input(type="uri_file"),
    output_metric: Output(type="uri_folder"),
):
    lines = [
        f"model: {model}",
        f"silo: {silo}",
        f"test data: {valid_data}",
        f"score output path: {output_metric}",
    ]

    for line in lines:
        print(line)

    (Path(output_metric) / "metric").write_text(silo)

def get_ml_client():
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()
    # credential = DefaultAzureCredential()
    client = MLClient(
        credential=credential,
        subscription_id="b8c23406-f9b5-4ccb-8a65-a8cb5dcd6a5a",
        resource_group_name="rge2etests",
        workspace_name="wse2etests",
    )
    return client

# Get a handle to workspace
ml_client = get_ml_client()

# Retrieve an already attached Azure Machine Learning Compute.
cluster_name = "cpucluster"
print(ml_client.compute.get(cluster_name))

@pipeline()
def pipeline_component_func(
    input_silos: Input(type="uri_file"),
    valid_data: Input(type="uri_file"),
):
    train = train_model(silo='s')
    valid = validate(model=train.outputs.output_model, silo='s', valid_data=valid_data)
    
    return {
        # "output_model": train.outputs.output_model,
        "output": valid.outputs.output_metric,
    }

# pipeline_component = ml_client.components.create_or_update(pipeline_component_func)
# By default, use time stamp as version: azureml:/subscriptions/b8c23406-f9b5-4ccb-8a65-a8cb5dcd6a5a/resourceGroups/rge2etests/providers/Microsoft.MachineLearningServices/workspaces/wse2etests/components/pipeline_component_func/versions/2022-11-12-05-57-55-4843396
pipeline_component = ml_client.components.create_or_update(pipeline_component_func, is_anonymous=True)
# set is_anonymous, use module entity id as version: azureml:/subscriptions/b8c23406-f9b5-4ccb-8a65-a8cb5dcd6a5a/resourceGroups/rge2etests/providers/Microsoft.MachineLearningServices/workspaces/wse2etests/components/azureml_anonymous/versions/1412164c-0302-484b-bf14-10a73d690c63
print(pipeline_component)