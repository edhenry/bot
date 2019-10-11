from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import shutil
import sys

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

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="1"

np.set_printoptions(precision=4)

def main():
    parser = argparse.ArgumentParser(description='Model Trainer')
    parser.add_argument('--input_dir', help="Directory containing training data (eg. /workspace/data)")
    parser.add_argument('--output_dir', help="Directory to save model to disk (eg. /tmp/model_dir)")
    parser.add_argument('--epochs', help="Number of training epochs (eg. 10)")
    parser.add_argument('--model_name', help="Name of output model (eg. lts)")
    parser.add_argument('--model_version', help="Version of the output model (eg. 001)")
    parser.add_argument('--data_augment', help="Enable or disable data augmentation (eg. True or False)")
    parser.add_argument('--subtract_pixel_mean', help="Enable or disable subtracting pixel mean from input images")
    parser.add_argument('--batch_size', help="Batch size for training data (eg. 128 (default))")
    args = parser.parse_args()

    print(f"Model Output Directory : {args.output_dir}/{args.model_name}")

    model_directory = os.path.join(args.output_dir, args.model_name)
    if os.path.isdir(model_directory):
        shutil.rmtree(model_directory)
    os.mkdir(model_directory)
    os.mkdir(os.path.join(model_directory, args.model_version))

    image_feature_description = {
        'timestamp': tf.io.FixedLenFeature([], tf.float32),
        'image': tf.io.FixedLenFeature([], tf.string),
        'steering_theta': tf.io.FixedLenFeature([], tf.float32),
        'accelerator': tf.io.FixedLenFeature([], tf.float32),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'capture_height': tf.io.FixedLenFeature([], tf.int64),
        'capture_width': tf.io.FixedLenFeature([], tf.int64),
        'capture_fps': tf.io.FixedLenFeature([], tf.int64),
        'num_channels': tf.io.FixedLenFeature([], tf.int64)
    }

    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = 128
    epochs = int(args.epochs)
    data_augmentation = args.data_augment

    def parse_image_function(example_proto):
        """Parse TFRecords into images
        
        Arguments:
            example_proto {[type]} -- Example protocol buffer defn
        """
        return tf.io.parse_single_example(example_proto, image_feature_description)
    
    def parse_dataset(dataset: tf.data.TFRecordDataset):
        """Parse entire TFRecord Dataset
        
        Arguments:
            dataset {tf.data.TFRecordDataset} -- dataset loaded from target filesystem
        """
        return dataset.map(parse_image_function)

    def load_data(directory: str):
        """
        Load TFRecords from target file system.
        
        Arguments:
            directory {str} -- Location on the target filesystem of the TFRecords to be processed
        """
        filenames = [(directory + '/' + f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=4)

        parsed_dataset = parse_dataset(dataset)
        return parsed_dataset

    def raw_to_numpy(dataset: tf.data.Dataset):
        """Convert string to numpy array
        
        Arguments:
            image {[type]} -- [description]
        """
        raw_img = dataset['image'].numpy()
        array_str = np.frombuffer(raw_img, np.uint8).reshape(224, 224, 3)
        dataset['image'] = array_str
        return dataset

    def deserialize_images(dataset: tf.data.Dataset):
        """Convert dataset['image'] back into numpy array
        
        Arguments:
            dataset {tf.data.Dataset} -- dataset to process
        """
        return dataset.map(raw_to_numpy)
    
    def plot_batch_sizes(ds):
        batch_sizes = [batch['steering_theta'].shape[0] for batch in ds]
        plt.bar(range(len(batch_sizes)), batch_sizes)
        plt.xlabel('Batch Number')
        plt.ylabel('Batch Size')
        plt.show()
    
    def plot_values(ds: tf.data.Dataset, key: str, title: str):
        """Plot steering values
        
        Arguments:
            ds {tf.data.Dataset} -- Dataset containing the training data collected from our car
            key {str} -- Feature contained within dataset that one would like to graph
        """
        value_dict = {}
        i=0
        for record in ds:
            value_dict[i] = record[key].numpy()
            i+=1
        x, y = zip(*value_dict.items())
        plt.figure(figsize=(10,7))
        plt.bar(x, y)
        plt.xlabel('Frame')
        plt.ylabel(key)
        plt.title(title)
        plt.savefig(f'{title}_distribution.png')

    raw_dataset = load_data(args.input_dir)

    plot_values(raw_dataset, 'accelerator', 'full_dataset_accelerator')
    plot_values(raw_dataset, 'steering_theta', 'full_dataset_steering_theta')

    # quick hack to get the length of the entire dataset for creating train / val / test splots
    record_count = 0
    for record in raw_dataset:
        record_count += 1

    TRAIN_SIZE = int(0.7 * record_count)
    VALIDATION_SIZE = int(0.15 * record_count)
    TEST_SIZE = int(0.15 * record_count)

    
    # Currently we shuffle, but may want to do planning in the future given
    # the sequence of the images
    full_dataset = raw_dataset.shuffle()
    train_dataset = full_dataset.take(TRAIN_SIZE)
    test_dataset = full_dataset.skip(TRAIN_SIZE)
    validation_dataset = test_dataset.take(VALIDATION_SIZE)
    test_dataset = test_dataset.take(TEST_SIZE)

    plot_values(train_dataset, 'steering_theta', 'training_dataset_steering_theta')
    plot_values(validation_dataset, 'steering_theta', 'test_dataset_steering_theta')
    plot_values(test_dataset, 'steering_theta', 'training_dataset_steering_theta')
    

    raw_dataset = raw_dataset.batch(int(args.batch_size), drop_remainder=True)

if __name__ == "__main__":
    main()