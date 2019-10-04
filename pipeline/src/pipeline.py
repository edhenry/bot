import kfp.dsl as dsl
import datetime
import os
from kubernetes import client as k8s_client

def PreprocessOp(name, input_dir, output_dir):
    return dsl.ContainerOp(
        name=name,
        image='edhenry/botcar-preprocess:latest',
        arguments=[
            '--input_dir', input_dir,
            '--output_dir', output_dir,
        ],
        file_outputs={'output': '/output.txt'}
    )

@dsl.pipeline(
    name='bot_learn2steer_pipeline',
    description='Demonstrate and end to end pipline for enabling an RC car to learn to steer'
)

def bot_training_pipeline(
    raw_data_dir='/mnt/workspace/raw_data',
    processed_data_dir='/mnt/workspace/processed_data',
):

    persistent_volume_name = 'lts-bot-data-claim'
    persistent_volume_path = '/mnt/workspace'

    op_dict = {}

    op_dict['preprocess'] = PreprocessOp('preprocess', raw_data_dir, processed_data_dir)

    for _, container_op in op_dict.items():
        container_op.add_volume(k8s_client.V1Volume(persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(claim_name=persistent_volume_name), name=persistent_volume_name)),

        container_op.add_volume_mount(k8s_client.V1VolumeMount(
            mount_path=persistent_volume_path,
            name=persistent_volume_name
        ))

if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(bot_training_pipeline, __file__ + '.tar.gz')