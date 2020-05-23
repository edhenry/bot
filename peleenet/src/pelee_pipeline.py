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

def Volume(name: str, action: str, resource_name: str,
                 volume_name: str, modes: str, size: str) -> dsl.VolumeOp:

                 return dsl.VolumeOp(
                     name=name,
                     action=action,
                     resource_name=resource_name,
                     volume_name=volume_name,
                     modes=modes,
                     size=size
                 )

def CheckVolume(name: str, image: str, command: List,
                arguments: List, pvolumes: dict) -> dsl.ContainerOp:

                return dsl.ContainerOp(
                    name=name,
                    image=image,
                    command=command,
                    arguments=arguments,
                    pvolumes=pvolumes
                )

def TrainingOp(name: str, input_dir: str, output_dir: str,
               epochs: int, model_name: str, model_version: int,
               batch_size: int, learning_rate: float, momentum: float,
               lr_patience: int, resize: int, scale_img: int, 
               dropout: int, crop_pct: float, growth_rate: int,
               num_classes: int, input_size: int, prefetch_size: int, 
               shuffle_buffer: int, volume: str):
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
        lr_patience {int} -- Patience interval to wait for 
        dropout {float} -- Percentage of dropout to add to the network (eg .5 == 50%)
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
        volume {str} -- Name of volume to map into container for access to data
    """

    return dsl.ContainerOp(
        name=name,
        # TODO enter container image name
        image='edhenry/peleenet-train:latest',
        arguments=[
            '--input_dir', input_dir,
            '--output_dir', output_dir,
            '--epochs', epochs,
            '--model_name', model_name,
            '--model_version', model_version,
            '--batch_size', batch_size,
            '--learning_rate', learning_rate,
            '--momentum', momentum,
            '--lr_patience', lr_patience,
            '--dropout', dropout,
            #'--resume_training', resume_training,
            '--resize', resize,
            '--scale_img', scale_img,
            '--crop_pct', crop_pct,
            #'--dataset_split', dataset_split,
            '--growth_rate', growth_rate,
            #'--bottle_neck_width', bottle_neck_width,
            '--num_classes', num_classes,
            '--input_size', input_size,
            '--prefetch_size', prefetch_size,
            '--shuffle_buffer', shuffle_buffer 
        ],
        pvolumes=volume,
        file_outputs={}
    ).set_gpu_limit(1)


@dsl.pipeline(
    name="PeleeNet",
    description="Model training pipeline for PeleeNet Model."
)

def peleenet_training_pipeline(
        raw_data_dir: str = '',
        processed_data_dir: str = '',
        input_dir: str = '',
        output_dir: str = '',
        epochs: int = 100,
        model_name: str = 'peleenet',
        model_version: int = 1,
        batch_size: int = 128,
        learning_rate: float = 0.4,
        momentum: float = 0.9,
        lr_patience: int = 2,
        dropout: float = 0.5,
#        resume_training: bool = False,
        resize: int = 32,
        scale_img: int = 7,
        crop_pct: float = 0.5,
#        dataset_split: List = [0.7, 0.15, 0,15],
        growth_rate: int = 32,
        num_classes: int = 1000,
        input_size: int = 32,
        prefetch_size: int = 2,
        shuffle_buffer: int = 1000,
        #bottle_neck_width: str = "1 2 4 4"
    ):

        volume = Volume(name="volume_creation", action="create", resource_name="kubeflow-test-pvc", 
                        volume_name="local-datasets-pipeline", modes=dsl.VOLUME_MODE_RWM, size="50Gi").set_display_name("Create PVC")

        check_volume = CheckVolume(name="check_vop", image="library/bash:4.4.23", command=['sh', '-c'], 
                                   arguments=['ls -lah /mnt'], pvolumes={'/mnt': volume.volume}).after(volume).set_display_name("Check Volume Exists")

        train = TrainingOp("train", input_dir=raw_data_dir, output_dir=output_dir,
                                      epochs=epochs, model_name=model_name, model_version=model_version,
                                      batch_size=batch_size, learning_rate=learning_rate,
                                      momentum=momentum, lr_patience=lr_patience, dropout=dropout, 
                                      resize=resize, scale_img=scale_img, crop_pct=crop_pct, 
                                      growth_rate=growth_rate, num_classes=num_classes, input_size=input_size, 
                                      prefetch_size=prefetch_size, shuffle_buffer=shuffle_buffer, volume={"/mnt": volume.volume}).after(check_volume).set_display_name("Trainer")

        #del_vol = volume.delete().after(train).set_display_name("Delete Volume")

if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(peleenet_training_pipeline, __file__ + '.tar.gz')