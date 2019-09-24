import kfp.dsl as dsl
import datetime
import os
from kubernetes import client as k8s_client

def PreprocessOp(name, input_dir, output_dir):
    return dsl.ContainerOp(
        name=name,
        image='nvcr.io/nvidia/tensorflow:19.08-py3',
        arguments=[
            '--input_dir', input_dir,
            '--output_dir', output_dir,
        ],
        file_outputs={'output': '/output.txt'}
    )