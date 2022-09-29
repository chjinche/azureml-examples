from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

from azure.ai.ml import MLClient, Input
from azure.ai.ml.dsl import pipeline
from pathlib import Path

from src.components import train_model, validate

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

@pipeline(default_compute=cluster_name)
def resolved_subgraph(
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

pipeline_component = ml_client.components.create_or_update(resolved_subgraph)
print(pipeline_component.id)

# /subscriptions/b8c23406-f9b5-4ccb-8a65-a8cb5dcd6a5a/resourceGroups/rge2etests/providers/Microsoft.MachineLearningServices/workspaces/wse2etests/components/resolved_subgraph/versions/1