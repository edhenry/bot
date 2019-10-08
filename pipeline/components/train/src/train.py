from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import shutil

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import (LearningRateScheduler, ModelCheckpoint,
                             ReduceLROnPlateau)
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Conv2D, Dense, Flatten, Input)
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants, tag_constants
from tensorflow.python.saved_model.signature_def_utils import \
    predict_signature_def


def main():
    parser = argparse.ArgumentParser(description='Model Trainer')
    parser.add_argument('--input_dir', help="Directory containing training data (eg. /workspace/data)")
    parser.add_argument('output_dir', help="Directory to save model to disk (eg. /tmp/model_dir)")
    parser.add_argument('--epochs', help="Number of training epochs (eg. 10)")
    parser.add_argument('--model_name', help="Name of output model (eg. lts)")
    parser.add_argument('--model_version', help="Version of the output model (eg. 001)")
    parser.add_argument('--data_augment', help="Enable or disable data augmentation (eg. True or False)")
    parser.add_argument('--subtract_pixel_mean', help="Enable or disable subtracting pixel mean from input images")
    args = parser.parse_args()

    print(args.output_dir, args.model_name)

    model_directory = os.path.join(args.output_dir, args.model_name)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(os.path.join(model_dir, args.model_version))

    batch_size = 128 
    epochs = int(args.epochs)
    data_augmentation = args.data_augment

    def load_data(directory: str):
        """
        Load TFRecords and divide into training / validation / testing datasets
        
        Arguments:
            directory {str} -- Location on the target filesystem of the TFRecords to be processed.
        """
        