import os
# enable private features
os.environ["AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED"] = "True"
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from pathlib import Path

try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()

# Get a handle to workspace
# ml_client = MLClient.from_config(credential=credential, path=Path(r"D:\Projects\vienna\src\aether\platform\backendV2\scripts\dcm_tests\orchestrator\configs\chjinche_canary.json"))
ml_client = MLClient.from_config(credential=credential, path=Path(r"D:\Projects\vienna\src\aether\platform\backendV2\scripts\dcm_tests\orchestrator\configs\wse2etests_master.json"))

# Retrieve an already attached Azure Machine Learning Compute.
# cluster_name = "cpu0829"
cluster_name = "cpucluster"
print(ml_client.compute.get(cluster_name))

from components import gen_silos, dynamic_subgraph, dummy, single_output_condition_func
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import Input, MLClient, UserIdentityConfiguration


@pipeline
def dynamic_subgraph_e2e(silos, valid_data, address):
    silos_node = gen_silos(params=silos)
    condition_node = single_output_condition_func(
        address=address
    )
    subgraph_node = dynamic_subgraph(input_silos=silos_node.outputs.output, valid_data=valid_data) #, cond=condition_node.outputs.output)
    subgraph_node.identity = UserIdentityConfiguration()
    dummy(input=subgraph_node.outputs.output)
    

pipeline_job = dynamic_subgraph_e2e(
    silos="silo1,silo2,silo3",
    valid_data=Input(path="wasbs://demo@dprepdata.blob.core.windows.net/Titanic.csv", type="uri_file"),
    address="h")

# Set pipeline level compute
pipeline_job.settings.default_compute = cluster_name

ml_client.jobs.validate(pipeline_job)

# Set experiment name and submit pipeline
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="dynamic_e2e_test_1114_graphId"
)

# show detail information of job
print(pipeline_job)
