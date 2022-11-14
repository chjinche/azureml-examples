from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from pathlib import Path

from pathlib import Path
from uuid import uuid4
import json

# mldesigner package contains the command_component which can be used to define component from a python function
from mldesigner import command_component, Input, Output
from mldesigner.dsl import dynamic, condition

ENVIRONMENT_DICT = dict(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file={
            "name": "default_environment",
            "channels": ["defaults"],
            "dependencies": [
                "python=3.8.12",
                "pip=21.2.2",
                {
                    "pip": [
                        "--extra-index-url=https://pkgs.dev.azure.com/azure-sdk/public/_packaging/azure-sdk-for-python/pypi/simple/",
                        "--extra-index-url=https://azuremlsdktestpypi.azureedge.net/test-sdk-cli-v2",
                        "mldesigner==0.0.76165727",
                        "mlflow==1.29.0",
                        "azureml-mlflow==1.45.0",
                        "azure-ai-ml==1.1.0a20221028006",
                        "azure-core==1.26.0",
                        "azure-common==1.1.28",
                        "azureml-core==1.45.0.post2",
                        "azure-ml-component==0.9.13.post1",
                        "azure-identity==1.11.0"
                    ]
                },
            ],
        }
    )


@command_component(environment=ENVIRONMENT_DICT)
def gen_silos(
    params: str,
    output: Output(type="uri_file"),
):
    with open(Path(output), "w") as fout:
        json.dump(params.split(','), fout)


@command_component(environment=ENVIRONMENT_DICT)
def single_output_condition_func(
        address: str
) -> Output(type="boolean", is_control=True):
    # validate the address is https or not
    result = address.startswith('https://')
    return result


@command_component(environment=ENVIRONMENT_DICT)
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


@command_component(environment=ENVIRONMENT_DICT)
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


@command_component(environment=ENVIRONMENT_DICT)
def dummy(
    input: Input(type="uri_folder"),
):
    print((Path(input) / "metric").read_text())


@pipeline(environment=ENVIRONMENT_DICT)
def pipeline_component_func(
    input_silos: Input(type="uri_file"),
    valid_data: Input(type="uri_file"),
    # cond: Input(type="boolean"),
):
    # condition(
    #     condition=cond,
    #     true_block=train_model(silo="true"),
    #     false_block=train_model(silo="false"),
    # )

    train = train_model(silo='s')
    valid = validate(model=train.outputs.output_model, silo='s', valid_data=valid_data)
    
    # return {
    #     # "output_model": train.outputs.output_model,
    #     "output": valid.outputs.output_metric,
    # }

    condition_node = single_output_condition_func(
        address="h"
    )

    return {
        # "output_model": train.outputs.output_model,
        "output": valid.outputs.output_metric,
        # "output": condition_node.outputs.output,
    }


@dynamic(environment=ENVIRONMENT_DICT)
def dynamic_subgraph(input_silos: Input(type="uri_file"), valid_data: Input(type="uri_file")) -> Output(type="uri_folder"):
# def dynamic_subgraph(input_silos: Input(type="uri_file"), valid_data: Input(type="uri_file"), cond: Input(type="boolean", is_control=True)) -> Output(type="uri_folder"):
# def dynamic_subgraph(input_silos: Input(type="uri_file"), valid_data: Input(type="uri_file"), cond: Input(type="boolean")) -> Output(type="boolean", is_control=True):
    # condition(
    #     condition=cond,
    #     true_block=train_model(silo="true"),
    #     false_block=train_model(silo="false"),
    # )

    with open(input_silos.result()) as fin:
        silos = json.load(fin)

    for silo in silos:
        train = train_model(silo=silo)
        valid = validate(model=train.outputs.output_model, silo=silo, valid_data=valid_data)
    
    condition_node = single_output_condition_func(
        address="h"
    )

    return {
        # "output_model": train.outputs.output_model,
        "output": valid.outputs.output_metric,
        # "output": condition_node.outputs.output,
    }
