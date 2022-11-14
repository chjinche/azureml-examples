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

from components import pipeline_component_func

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

# pipeline_component = ml_client.components.create_or_update(pipeline_component_func)
# By default, use time stamp as version: azureml:/subscriptions/b8c23406-f9b5-4ccb-8a65-a8cb5dcd6a5a/resourceGroups/rge2etests/providers/Microsoft.MachineLearningServices/workspaces/wse2etests/components/pipeline_component_func/versions/2022-11-12-05-57-55-4843396
pipeline_component = ml_client.components.create_or_update(pipeline_component_func, is_anonymous=True)
# set is_anonymous, use module entity id as version: azureml:/subscriptions/b8c23406-f9b5-4ccb-8a65-a8cb5dcd6a5a/resourceGroups/rge2etests/providers/Microsoft.MachineLearningServices/workspaces/wse2etests/components/azureml_anonymous/versions/1412164c-0302-484b-bf14-10a73d690c63
print(pipeline_component)