import datetime
import os

from typing import List
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
        image='edhenry/peleenet-train:latest',
        arguments=[
            '--input_dir', input_dir,
            '--output_dir', output_dir,
        ],
        file_outputs={}
    )

def TrainingOp(name: str, input_dir: str, output_dir: str,
               epochs: int, model_name: str, model_version: int,
               batch_size: int, learning_rate: float, momentum: float,
               resume_training: bool, resize: int, scale_img: int,
               crop_pct: float, dataset_split: list, growth_rate: int,
               bottle_neck_width: List[int], num_classes: int, input_size: int,
               prefetch_size: int, shuffle_buffer: int):
    """Start model training within Kubeflow pipeline
    
    Arguments:
        name {str} -- operation name for Kubeflow UID (eg Training)
        output_dir {str} -- Output directory containing artifacts from training (eg. "/directory/on/local/filesystem")
        epochs {int} -- Number of epochs for model training (eg. 10)
        model_name {str} -- Name of the model (eg "peleenet")
        model_version {int} -- Version of the model (eg 1)
        batch_size {int} -- Batch size to use for mini-batch training (eg. 64)
        learning_rate {float} -- Learning rate for training model (eg. 1e-3 or 0.0001)
        momentum {float} -- Momentum factor for use with SGD optimizer
        resume_training {bool} -- Resume training of a saved model (eg. True or False)
        resize {int} -- Resize training data (eg. 32 (where original image size is (224,224) this would resize the image to (256, 256)))
        scale_img {int} -- Factor by which to scale the input image (eg. 7 (if the input image is 32x32x3 (HWC) the output would be (224,224,3)))
        crop_pct {float} -- Percentage to center crop training images (eg. 0.5 will center crop to the middle 50% of pixels in the image)
        dataset_split {list} -- Splits to use for Training, Validation, and Test sets (if applicable)
        growth_rate {int} -- Growth rate to use (see DenseNet and PeleeNet paper : https://arxiv.org/abs/1804.06882)
        bottle_neck_width {List[int]} -- Bottle beck widths to use for the Dense layers
        num_classes {int} -- Number of classes the model is being used for
        input_size {int} -- Input size of the images used for training
        prefetch_size {int} -- Number of batches to prefetch for training
        shuffle_buffer {int} -- Number of examples to store in buffer for shuffling datasets too large to fit in memory
    """
    return dsl.ContainerOp(
        name=name,
        # TODO enter container image name
        image='edhenry/peleenet-train:latest',
        arguments=[
            '--input_dir', input_dir,
            '--ouput_dir', output_dir,
            '--epochs', epochs,
            '--model_name', model_name,
            '--batch_size', batch_size,
            '--learning_rate', learning_rate,
            '--momentum', momentum,
            '--resume_training', resume_training,
            '--resize', resize,
            '--scale_img', scale_img,
            '--crop_pct', crop_pct,
            '--dataset_split', dataset_split,
            '--growth_rate', growth_rate,
            '--bottle_neck_width', bottle_neck_width,
            '--num_classes': num_classes,
            '--input_size': input_size,
            '--prefetch_size': prefetch_size,
            '--shuffle_buffer': shuffle_buffer 
        ],
        file_outputs={}
    ).set_gpu_limit(1)

@dsl.pipeline(
    name="PeleeNet",
    description="Model training pipeline for PeleeNet Model."
)

def peleenet_training_pipeline(
        raw_data_dir='',
        processed_data_dir='',
        output_dir='',
        epochs=100,
        model_name='peleenet',
        model_version=1,
        batch_size=128,
        learning_rate=0.4,
        momentum=0.9,
        resume_training=False,
        resize=32,
        scale_img=7,
        crop_pct=0.5,
        dataset_split=[0.7, 0.15, 0,15],
        growth_rate=32,
        num_classes=1000,
        input_size=32,
        prefetch_size=2,
        shuffle_buffer=1000
    ):
    
        persistent_volume_name = 'lts-bot-data-claim'
        persistent_volume_path = '/mnt/workspace'

        op_dict = {}

        #op_dict['preprocess'] = PreprocessOp('preprocess', raw_data_dir, processed_data_dir)
        op_dict['train'] = TrainingOp('train', input_dir=raw_data_dir, output_dir=output_dir,
                                  epochs=epochs, model_name=model_name, model_version=model_version,
                                  batch_size=batch_size, learning_rate=learning_rate, momentum=momentum,
                                  resume_training=resume_training, resize=resize, scale_img=scale_img,
                                  crop_pct=crop_pct, dataset_split=dataset_split, growth_rate=growth_rate,
                                  bottle_neck_width=bottle_neck_width, num_classes=num_classes, input_size=input_size,
                                  prefetch_size=prefetch_size, shuffle_buffer=shuffle_buffer)

        # for _, container_op in op_dict.items():
        #     container_op.add_volume(k8s_client.V1Volume(persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(claim_name=persistent_volume_name), name=persistent_volume_name)),

        #    container_op.add_volume_mount(k8s_client.V1VolumeMount(
        #         mount_path=persistent_volume_path,
        #         name=persistent_volume_name
        #     ))

if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(peleenet_training_pipeline, __file__ + '.tar.gz')