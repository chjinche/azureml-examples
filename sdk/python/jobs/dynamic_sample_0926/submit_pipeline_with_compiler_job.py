# import required libraries
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

from azure.ai.ml import MLClient, Input
from azure.ai.ml.dsl import pipeline

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

from src.components import gen_silos, train_model, validate, dynamic

cluster_name = "cpucluster"
# define a pipeline with component
@pipeline(default_compute=cluster_name)
def pipeline_with_compiler_job(silos, valid_data):
    silos = gen_silos(params=silos)
    dynamic_compiler = dynamic(silo_input=silos.outputs.output, valid_data=valid_data)


pipeline_job = pipeline_with_compiler_job(
    silos="silo1,silo2,silo3",
    valid_data=Input(
        path="wasbs://demo@dprepdata.blob.core.windows.net/Titanic.csv", type="uri_file"
    ),
)

# submit job to workspace
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="pipeline_with_compiler_job"
)
print(pipeline_job)