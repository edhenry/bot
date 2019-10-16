from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import shutil
import sys
import cv2
from typing import Tuple

import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
# from keras import backend as K
# from keras.callbacks import (LearningRateScheduler, ModelCheckpoint,
#                              ReduceLROnPlateau)
# from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
#                           Conv2D, Dense, Flatten, Input)
# from keras.models import Model, load_model
# from keras.optimizers import Adam
# from keras.preprocessing.image import ImageDataGenerator
# from keras.regularizers import l2
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
        parsed_data = tf.io.parse_single_example(example_proto, image_feature_description)
        # image = tf.io.decode_raw(parsed_data['image'], tf.int8)
        image = tf.io.parse_tensor(parsed_data['image'], out_type=tf.int8)
        image = tf.reshape(image, [224,224,3])
        steering_theta = tf.cast(parsed_data['steering_theta'], tf.float32)
        accelerator = tf.cast(parsed_data['accelerator'], tf.float32)
        return image, steering_theta, accelerator
    
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

    # plot_values(train_dataset, 'steering_theta', 'training_dataset_steering_theta')
    # plot_values(validation_dataset, 'steering_theta', 'validation_dataset_steering_theta')
    # plot_values(test_dataset, 'steering_theta', 'test_dataset_steering_theta')

    train_batch = train_dataset.batch(int(args.batch_size), drop_remainder=True)

    # # Model defn
    # inputs = tf.keras.Input(shape=(150528,), name='image')
    # #x = layers.Dense(4096, activation='relu')(inputs)
    # x = layers.Dense(2048, activation='relu')(inputs)
    # x = layers.Dense(1024, activation='relu')(x)
    # x = layers.Dense(128, activation='relu')(x)
    # x = layers.Dense(64, activation='relu')(x)
    # outputs = layers.Dense(1, activation='tanh', name='outputs')(x)

    inputs = tf.keras.Input(shape=(150528,), name='image')
    x = tf.keras.layers.Conv2D(24, (5,5), name='conv1', strides=(2,2), padding='valid', activation='relu', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.Conv2D(36, (5,5), name='conv2', strides=(2,2), padding='valid', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Conv2D(48, (5,5), name='conv3', strides=(2,2), padding='valid', activation='relu', kernel_initializer='he_normal')(x)

    x = layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv2D(64, (3,3), name='conv4', strides=(1,1), padding='valid', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Conv2D(64, (3,3), name='conv5', strides=(1,1), padding='valid', activation='relu', kernel_initializer='he_normal')(x)

    x = tf.keras.layers.Flatten(name='flatten')(x)

    x = layers.Dense(100, name='fc1', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dense(50, name='fc2', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dense(10, name='fc3', activation='relu', kernel_initializer='he_normal')(x)
    outputs = layers.Dense(1, name='output', activation='tanh', kernel_initializer='he_normal')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    criterion = tf.losses.MeanSquaredError()
    optimizer = tf.optimizers.Adam(learning_rate=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    print(model.summary())

    for epoch in range(int(args.epochs)):
        for step, (x, y, z) in enumerate(train_batch):
            with tf.GradientTape() as tape:
                logits = model(x)
                logits = tf.squeeze(logits, axis=1)
                loss = criterion(y, logits)
            
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        print(epoch, 'loss:', loss.numpy())

if __name__ == "__main__":
    main()