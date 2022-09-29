# import required libraries
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

from azure.ai.ml import MLClient, Input
from azure.ai.ml.dsl import pipeline
import os

try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()

# Get a handle to workspace
ml_client = MLClient.from_config(credential=credential)

# Retrieve an already attached Azure Machine Learning Compute.
cluster_name = "cpucluster"
print(ml_client.compute.get(cluster_name))

from src.components import gen_silos, train_model, validate

os.environ['AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED'] = "true"

@pipeline(default_compute=cluster_name)
def mock_subgraph(
    silo_input: Input(type="uri_folder"),
    valid_data: Input(type="uri_file"),
    param: str = "123"
):
    train = train_model(silo='silo1')
    valid = validate(model=train.outputs.output_model, silo='silo1', valid_data=valid_data)
    return {
        "output_model": train.outputs.output_model,
        "output_metric": valid.outputs.output_metric,
    }


cluster_name = "cpucluster"
# define a pipeline with component
@pipeline(default_compute=cluster_name)
def pipeline_with_subgraph(silo_str, valid_data):
    silos = gen_silos(params=silo_str)
    subgraph = mock_subgraph(silo_input=silos.outputs.output, valid_data=valid_data)


pipeline_job = pipeline_with_subgraph(
    silo_str="silo1,silo2,silo3",
    valid_data=Input(
        path="wasbs://demo@dprepdata.blob.core.windows.net/Titanic.csv", type="uri_file"
    ),
)

# submit job to workspace
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="pipeline_with_subgraph"
)
print(pipeline_job)
