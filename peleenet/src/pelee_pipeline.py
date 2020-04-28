import datetime
import os

import kfp.dsl as dsl
from kubernetes import client as k8s_client

def PreprocessOp(name, input_dir, output_dir):
    """Preprocessing Op for Kubeflow pipeline for CIFAR 10/100 training for PeleeNet
    
    Arguments:
        name {[type]} -- [description]
        input_dir {[type]} -- [description]
        output_dir {[type]} -- [description]
    """
    return dsl.ContainerOp(
        name=name,
        # TODO enter container image name
        image='',
        arguments=[
            '--input_dir', input_dir,
            '--output_dir', outout_dir,
        ],
        file_outputs={}
    )

def TrainingOp(name: str, input_dir: str, output_dir: str,
               epochs: int, model_name: str, model_version: int,
               batch_size: int, learning_rate: float, resume_training: bool):
    """Start model training within Kubeflow pipeline
    
    Arguments:
        name {str} -- operation name for Kubeflow UID (eg Training)
        input_dir {str} -- Input directory containing the training data (eg. "/dir/on/local/filesystem")
        output_dir {str} -- Output directory containing artifacts from training (eg. "/directory/on/local/filesystem")
        epochs {int} -- Number of epochs for model training (eg. 10)
        model_name {str} -- Name of the model (eg "peleenet")
        model_version {int} -- Version of the model (eg 1)
        batch_size {int} -- Batch size to use for mini-batch training (eg. 64)
        learning_rate {float} -- Learning rate for training model (eg. 1e-3 or 0.0001)
        resume_training {bool} -- Resume training of a saved model (eg. True or False)
    """
    return dsl.ContainerOp(
        name=name,
        # TODO enter container image name
        image='',
        arguments=[
            '--input_dir', input_dir,
            '--ouput_dir', output_dir,
            '--epochs', epochs,
            '--model_name', model_name,
            '--batch_size', batch_size,
            '--learning_rate', learning_rate,
            '--resume_training', resume_training
        ],
        file_outputs={}
    )#.set_gpu_limit(1)

    def pelee_net_training_pipeline(
        raw_data_dir='',
        processed_data_dir='',
        output_dir='',
        epochs=100,
        model_name='botcar',
        model_version=1,
        batch_size=64,
        learning_rate=0.0001,
        resume_training=False
    ):
    