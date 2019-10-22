from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import os
import pathlib
import shutil
import sys
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
    parser.add_argument('--learning_rate', help="Learning rate to use with optimizer (eg. 0.0001 or 1e-4)")
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
        batch_size = int(args.batch_size)
    else:
        batch_size = 128
    EPOCHS = int(args.epochs)
    LEARNING_RATE = float(args.learning_rate)
    data_augmentation = args.data_augment


    def parse_image_function(example_proto):
        """Parse TFRecords into images
        
        Arguments:
            example_proto {[type]} -- Example protocol buffer defn
        """
        parsed_data = tf.io.parse_single_example(example_proto, image_feature_description)
        image = tf.io.decode_raw(parsed_data['image'], tf.int8)
        # image = tf.io.parse_tensor(parsed_data['image'], out_type=tf.int8)
        image = tf.reshape(image, [224,224,3])
        image = tf.image.convert_image_dtype(image, tf.float32)
        steering_theta = tf.cast(parsed_data['steering_theta'], tf.float32)
        accelerator = tf.cast(parsed_data['accelerator'], tf.float32)
        return image, steering_theta
    
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
            value_dict[i] = record[1].numpy()
            i+=1
        x, y = zip(*value_dict.items())
        plt.figure(figsize=(10,7))
        plt.bar(x, y)
        plt.xlabel('Frame')
        plt.ylabel(key)
        plt.title(title)
        plt.savefig(f'{title}_distribution.png')

    raw_dataset = load_data(args.input_dir)

    # quick hack to get the length of the entire dataset for creating train / val / test splots
    record_count = 0
    for record in raw_dataset:
        record_count += 1

    print(f'Total Records contained within this directory : {record_count}')

    TRAIN_SIZE = int(0.7 * record_count)
    VALIDATION_SIZE = int(0.15 * record_count)
    TEST_SIZE = int(0.15 * record_count)

    
    # Currently we shuffle, but may want to do planning in the future given
    # the sequence of the images
    full_dataset = raw_dataset.shuffle(record_count)
    train_dataset = full_dataset.take(TRAIN_SIZE)
    test_dataset = full_dataset.skip(TRAIN_SIZE)
    validation_dataset = test_dataset.take(VALIDATION_SIZE)
    test_dataset = test_dataset.take(TEST_SIZE)

    train_batch = train_dataset.batch(batch_size, drop_remainder=True)
    test_batch = test_dataset.batch(batch_size, drop_remainder=True)
    validation_batch = validation_dataset.batch(batch_size, drop_remainder=True)

    # Model definition
    input_image = Input(shape=(224,224,3,), name='image')
    conv1 = Conv2D(24, (5,5), name='conv1', strides=(2,2), padding='valid', activation='relu', kernel_initializer='he_normal')(input_image)
    conv2 = Conv2D(36, (5,5), name='conv2', strides=(2,2), padding='valid', activation='relu', kernel_initializer='he_normal')(conv1)
    conv3 = Conv2D(48, (5,5), name='conv3', strides=(2,2), padding='valid', activation='relu', kernel_initializer='he_normal')(conv2)

    dropout_1 = Dropout(0.5)(conv3)

    conv4 = Conv2D(64, (3,3), name='conv4', strides=(1,1), padding='valid', activation='relu', kernel_initializer='he_normal')(dropout_1)
    conv5 = Conv2D(64, (3,3), name='conv5', strides=(1,1), padding='valid', activation='relu', kernel_initializer='he_normal')(conv4)

    flatten = Flatten(name='flatten')(conv5)

    # input_acceleration = Input(shape=(1,), name='acceleration_input')
    # acceleration = Dense(1, name='acceleration_layer', activation='tanh', kernel_initializer='he_normal')(input_acceleration)

    # concat = concatenate([flatten, acceleration])

    fc1 = Dense(100, name='fc1', activation='relu', kernel_initializer='he_normal')(flatten)
    fc2 = Dense(50, name='fc2', activation='relu', kernel_initializer='he_normal')(fc1)
    fc3 = Dense(10, name='fc3', activation='relu', kernel_initializer='he_normal')(fc2)
    output_steering = Dense(1, name='steering_output', activation='tanh', kernel_initializer='he_normal')(fc3)
    #output_acceleration = Dense(1, name='acceleration_output', activation='tanh', kernel_initializer='he_normal')(fc3)

    model = keras.Model(inputs=input_image, outputs=output_steering)

    plot_model(model, f'{args.model_name}_{args.model_version}_model.png', show_shapes=True)

    steering_loss = tf.losses.MeanSquaredError()
    acceleration_loss = tf.losses.MeanSquaredError()
    optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    validation_loss = tf.keras.metrics.Mean(name='validation_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    model.compile(optimizer=optimizer,
                 loss=keras.losses.MeanSquaredError(),
                 metrics=['mse'])

    model.fit(train_batch, epochs=EPOCHS)

if __name__ == "__main__":
    main()
