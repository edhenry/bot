from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import datetime
import json
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
from tensorflow.keras import Input, callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
    parser.add_argument('--resume_training', help="Resume training of a model version that already exists (eg. True or False)")
    parser.add_argument('--par_reads', help="Number of processes to use for dataset parallelism")
    args = parser.parse_args()

    if args.batch_size:
        BATCH_SIZE = int(args.batch_size)
    else:
        BATCH_SIZE = 128
    EPOCHS = int(args.epochs)
    LEARNING_RATE = float(args.learning_rate)
    DATA_AUGMENTATION = args.data_augment
    MODEL_VERSION = int(args.model_version)
    MODEL_NAME = args.model_name
    PAR_READS = int(args.par_reads)

    print(f"Model Output Directory : {args.output_dir}/{MODEL_NAME}")
    print(f"Current Model Version : {MODEL_VERSION}")

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
        print(filenames)
        files = tf.data.Dataset.list_files(filenames)

        ## dataset paralleism
        ## Here we're utilizing parallelism to deserialize TFRecords from disk
        dataset = files.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=4)

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

    train_batch = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
    test_batch = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
    validation_batch = validation_dataset.batch(BATCH_SIZE, drop_remainder=True)

    # plot_values(train_dataset, 'steering_theta', 'training_dataset_steering_theta')
    # plot_values(validation_dataset, 'steering_theta', 'validation_dataset_steering_theta')
    # plot_values(test_dataset, 'steering_theta', 'test_dataset_steering_theta')

    class BotModel(Model):
        def __init__(self):
            super(BotModel, self).__init__()
            # self.input_layer = Input(shape=(224,224,3,), name='image')
            self.conv1 = Conv2D(24, (5,5), name='conv1', strides=(2,2), padding='valid', activation='relu', kernel_initializer='he_normal')
            self.conv2 = Conv2D(36, (5,5), name='conv2', strides=(2,2), padding='valid', activation='relu', kernel_initializer='he_normal')
            self.conv3 = Conv2D(48, (5,5), name='conv3', strides=(2,2), padding='valid', activation='relu', kernel_initializer='he_normal')

            self.dropout_1 = Dropout(0.5)

            self.conv4 = Conv2D(64, (3,3), name='conv4', strides=(1,1), padding='valid', activation='relu', kernel_initializer='he_normal')
            self.conv5 = Conv2D(64, (3,3), name='conv5', strides=(1,1), padding='valid', activation='relu', kernel_initializer='he_normal')

            self.flatten = Flatten(name='flatten')

            self.fc1 = Dense(100, name='fc1', activation='relu', kernel_initializer='he_normal')
            self.fc2 = Dense(50, name='fc2', activation='relu', kernel_initializer='he_normal')
            self.fc3 = Dense(10, name='fc3', activation='relu', kernel_initializer='he_normal')
            self.output_val = Dense(1, name='output', activation='tanh', kernel_initializer='he_normal')

        def call(self, x):
            # x = self.input_layer(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)

            x = self.dropout_1(x)
            x = self.conv4(x)
            x = self.conv5(x)

            x = self.flatten(x)

            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            x = self.output_val(x)
            return x

    loss_object = tf.losses.MeanSquaredError()
    optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    validation_loss = tf.keras.metrics.Mean(name='validation_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    print(f"Test loss: {test_loss}")

    model = BotModel()

    @tf.function
    def train_step(images: tf.data.Dataset.batch, labels: tf.data.Dataset.batch):
        """Training step for our model
        
        Arguments:
            images {tf.data.Dataset.batch} -- Batch input data for model training
            labels {tf.data.Dataset.batch} -- Batch label data for model training
        """
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
    
    @tf.function
    def validation_step(images: tf.data.Dataset.batch, labels: tf.data.Dataset.batch):
        """Validation step for our model
        
        Arguments:
            images {tf.data.Dataset.batch} -- Batch input data for model validation
            labels {tf.data.Dataset.batch} -- Batch label data for model validation
        """
        predictions = model(images)
        v_loss = loss_object(labels, predictions)

        validation_loss(v_loss)

    @tf.function
    def test_step(images: tf.data.Dataset.batch, labels: tf.data.Dataset.batch):
        """Test step
        
        Arguments:
            images {tf.data.Dataset.batch} -- Batch input data for model testing
            labels {tf.data.Dataset.batch} -- Batch label data for model testing
        """
        predictions = model(images)
        t_loss = loss_object(predictions, labels)

        test_loss(t_loss)

    # Make model directory structure on filesystem
    # TODO : Break this functionality out into a utils module
    model_directory = os.path.join(args.output_dir, args.model_name)
    if os.path.isdir(model_directory):
        pass
    else:
        os.mkdir(model_directory)
        
    if os.path.isdir(os.path.join(model_directory, str(MODEL_VERSION))):
        pass
    else:
        os.mkdir(os.path.join(model_directory, str(MODEL_VERSION)))

    checkpoint_dir = os.path.join(model_directory, str(MODEL_VERSION), "ckpt")
    tensorboard_dir = os.path.join(model_directory, str(MODEL_VERSION), "logs")
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        shutil.rmtree(tensorboard_dir)
    os.mkdir(checkpoint_dir)
    os.mkdir(tensorboard_dir)   

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_path = checkpoint_dir + '/'
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_path, max_to_keep=5)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # TODO clean up directory creation logic
    # have one root variable dir = model_directory + str(MODEL_VERSION) + 'logs/gradient_tape'
    train_log_dir = tensorboard_dir + '/gradient_tape/' + current_time + '/train'
    test_log_dir = tensorboard_dir + '/gradient_tape/' + current_time + '/test'
    print(f"Training Log Directory : {train_log_dir}")
    print(f"Test Log Directory : {test_log_dir}")
    validation_log_dir = 'logs/gradient_tape/' + current_time + '/validation'
    os.makedirs(train_log_dir)
    os.makedirs(validation_log_dir)
    os.makedirs(test_log_dir)


    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    validation_summary_writer = tf.summary.create_file_writer(validation_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for epoch in range(EPOCHS):
        print(f'Starting epoch number {(epoch + 1)}...')
        for step, (images, labels, _) in enumerate(train_batch):
            train_step(images, labels)
        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch)    
        
        for step, (images, labels, _) in enumerate(validation_batch):
            validation_step(images, labels)
        with validation_summary_writer.as_default():
            tf.summary.scalar('validation_loss', validation_loss.result(), step=epoch)
        
        print(f"Epoch : {epoch+1}, Training Loss : {train_loss.result()}, Validation Loss : {validation_loss.result()}")

        if epoch % 1 == 0:
            for test_images, test_labels, _ in test_batch:
                test_step(test_images, test_labels)
            with test_summary_writer.as_default():
                tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
            print(f"Epoch : {epoch+1}, Training Loss : {train_loss.result()}, Test Loss : {test_loss.result()}")
            checkpoint_manager.save(checkpoint_number=None)       
        
        train_loss.reset_states()
        validation_loss.reset_states()
        test_loss.reset_states()
    
    model.save((model_directory + "/" + MODEL_NAME + "-" + str(MODEL_VERSION) + "-" + str(EPOCHS)))

    tensorboard_metadata = {
      'outputs' : [{
        'type': 'tensorboard',
        'source': f"'{(tensorboard_dir + '/gradient_tape')}'",
      }]
    }

    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(tensorboard_metadata, f)    
    
    with open('/output.txt', 'w') as f:
        f.write(args.output_dir)

if __name__ == "__main__":
    main()
