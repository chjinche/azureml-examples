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

# init customer environment with conda YAML
# the YAML file shall be put under your code folder.
conda_env = dict(
    # note that mldesigner package must be included.
    conda_file=Path(__file__).parent / "conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
)

@command_component(
    display_name="dynamic",
    environment=conda_env,
    # specify your code folder, default code folder is current file's parent
    # code='.'
)
def dynamic(
    silo_input: Input(type="uri_folder"),
    valid_data: Input(type="uri_file"),
    output_model: Output(type="uri_folder"),
    output_metric: Output(type="uri_folder"),
    param: str = "123"
):
    # from azure.ai.ml.identity import AzureMLOnBehalfOfCredential

    # from azure.ai.ml import MLClient, Input
    # from azure.ai.ml.dsl import pipeline
    # from pathlib import Path

    # def get_ml_client():
    #     credential = AzureMLOnBehalfOfCredential()
    #     client = MLClient(
    #         credential=credential,
    #         subscription_id="b8c23406-f9b5-4ccb-8a65-a8cb5dcd6a5a",
    #         resource_group_name="rge2etests",
    #         workspace_name="wse2etests",
    #     )
    #     return client

    # # Get a handle to workspace
    # ml_client = get_ml_client()

    # # Retrieve an already attached Azure Machine Learning Compute.
    # cluster_name = "cpucluster"
    # print(ml_client.compute.get(cluster_name))

    # @pipeline()
    # def pipeline_component_func(
    #     silo_input: Input(type="uri_folder"),
    #     valid_data: Input(type="uri_file"),
    #     param: str = "123"
    # ):
    #     silo = (Path(silo_input) / "silos").read_text()
    #     chunks = silo.split(",")
    #     for s in chunks:
    #         train = train_model(silo=s)
    #         valid = validate(model=train.outputs.output_model, silo=s, valid_data=valid_data)
        
    #     return {
    #         "output_model": train.outputs.output_model,
    #         "output_metric": valid.outputs.output_metric,
    #     }
    
    # pipeline_component = ml_client.components.create_or_update(pipeline_component_func)
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        from mlflow.utils.rest_utils import http_request

    except ImportError as e:
        raise ImportError("mlflow is required to write control outputs. Please install mlflow first.") from e

    # step1: get mlflow run
    with mlflow.start_run() as run:
        client = MlflowClient()

        # step2: get auth
        cred = client._tracking_client.store.get_host_creds()

        # step3: update host to run history
        cred.host = cred.host.replace(
            "api.azureml.ms",
            "experiments.azureml.net",
        ).replace("mlflow/v1.0", "history/v1.0")

        # step4: call run history
        http_request(
            host_creds=cred,
            endpoint="/experimentids/{}/runs/{}".format(run.info.experiment_id, run.info.run_id),
            method="PATCH",
            json={
                "runId": run.info.run_id,
                "properties": {"subPipelineComponentId": "/subscriptions/b8c23406-f9b5-4ccb-8a65-a8cb5dcd6a5a/resourceGroups/rge2etests/providers/Microsoft.MachineLearningServices/workspaces/wse2etests/components/resolved_subgraph/versions/1"},
            },
        )
        # print(f"Finished writing to run properties: '{pipeline_component.id}'")
