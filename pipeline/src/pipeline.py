import datetime
import os

import kfp.dsl as dsl
from kubernetes import client as k8s_client


def PreprocessOp(name, input_dir, output_dir):
    return dsl.ContainerOp(
        name=name,
        image='edhenry/botcar-preprocess:latest',
        arguments=[
            '--input_dir', input_dir,
            '--output_dir', output_dir,
        ],
        file_outputs={}
    )

def TrainingOp(name: str, input_dir: str, output_dir: str,
               epochs: int, model_name: str, model_version: int, 
               batch_size: int, learning_rate: float, resume_training: bool):
    """Start training process as outlined in pipeline
    
    Arguments:
        name {str} -- operation name for Kubeflow UID (eg "Training")
        input_dir {str} -- Input directory containing training data (eg "/directory/on/local/filesystem")
        output_dir {str} -- Ouput directory where we store artifacts from training (eg "/directory/on/local/filesystem")
        epochs {int} -- Number of epochs for model training (eg "10")
        model_name {str} -- Name of the model (eg "bot_model")
        model_version {int} -- Version of the model (eg "1")
        batch_size {int} -- Batch size to use for mini-batch training (eg. "64")
        learning_rate {float} -- Learning rate for training model (eg "1e-3" or "0.0001")
        resume_training {bool} -- Resume training of an already trained model (eg "True" or "False")
    """
    return dsl.ContainerOp(
        name=name,
        image='edhenry/botcar-train:latest',
        arguments=[
            '--input_dir', input_dir,
            '--output_dir', output_dir,
            '--epochs', epochs,
            '--model_name', model_name,
            '--model_version', model_version,
            '--batch_size', batch_size,
            '--learning_rate', learning_rate,
            '--resume_training', resume_training
        ],
        file_outputs={}
    ).set_gpu_limit(1)

@dsl.pipeline(
    name='bot_learn2steer_pipeline',
    description='Demonstrate and end to end pipline for enabling an RC car to learn to steer'
)

def bot_training_pipeline(
    raw_data_dir='/mnt/workspace/raw_data',
    processed_data_dir='/mnt/workspace/processed_data',
    output_dir='/mnt/workspace/saved_model',
    epochs=100,
    model_name='botcar',
    model_version=1,
    batch_size=64,
    learning_rate=0.0001,
    resume_training=False
):

    persistent_volume_name = 'lts-bot-data-claim'
    persistent_volume_path = '/mnt/workspace'

    op_dict = {}

    op_dict['preprocess'] = PreprocessOp('preprocess', raw_data_dir, processed_data_dir)
    op_dict['train'] = TrainingOp('train', input_dir=raw_data_dir, output_dir=output_dir,
                                  epochs=epochs, model_name=model_name, model_version=model_version,
                                  batch_size=batch_size, learning_rate=learning_rate, resume_training=resume_training)

    for _, container_op in op_dict.items():
        container_op.add_volume(k8s_client.V1Volume(persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(claim_name=persistent_volume_name), name=persistent_volume_name)),

        container_op.add_volume_mount(k8s_client.V1VolumeMount(
            mount_path=persistent_volume_path,
            name=persistent_volume_name
        ))

if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(bot_training_pipeline, __file__ + '.tar.gz')
