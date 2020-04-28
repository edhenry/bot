import argparse
import json
import os
import pickle
import shutil
from collections import OrderedDict
import datetime
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (Activation, AveragePooling2D,
                                     BatchNormalization, Concatenate, Conv2D,
                                     Dense, Dropout, Flatten, Input, MaxPool2D)
from tensorflow.keras.models import Model

import tensorflow_datasets as tfds

class _DenseLayer(Model):
    def __init__(self, num_input_features, growth_rate, bottleneck_width, drop_rate):
        super(_DenseLayer, self).__init__()

        growth_rate = int(growth_rate / 2)
        inter_channel = int(growth_rate * bottleneck_width / 4) * 4

        if inter_channel > num_input_features / 2:
            inter_channel = int(num_input_features / 8) * 4
            print(f'adjusting inter_channel to {inter_channel}')
        
        self.branch1a = BasicConv2D(num_input_features, inter_channel, kernel_size=1, padding="same")
        self.branch1b = BasicConv2D(inter_channel, growth_rate, kernel_size=3, padding="same")

        self.branch2a = BasicConv2D(num_input_features, inter_channel, kernel_size=1, padding="same")
        self.branch2b = BasicConv2D(inter_channel, growth_rate, kernel_size=3, padding="same")
        self.branch2c = BasicConv2D(growth_rate, growth_rate, kernel_size=3, padding="same")

    def call(self, x):
        branch1 = self.branch1a(x)
        branch1 = self.branch1b(branch1)

        branch2 = self.branch2a(x)
        branch2 = self.branch2b(branch2)
        branch2 = self.branch2c(branch2)

        return Concatenate()([x, branch1, branch2])

class _DenseBlock(Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add(layer)


class _StemBlock(Model):
    def __init__(self, num_init_features):
        super(_StemBlock, self).__init__()

        num_stem_features = int(num_init_features/2)

        self.stem1 = BasicConv2D(out_channels=num_init_features, kernel_size=3, strides=2)
        self.stem2a = BasicConv2D(out_channels=num_stem_features, kernel_size=1, strides=1)
        self.stem2b = BasicConv2D(out_channels=num_init_features, kernel_size=3, strides=2)
        self.stem3 = BasicConv2D(out_channels=num_init_features, kernel_size=1, strides=1)
        self.pool = MaxPool2D(2)

    def call(self, x):
        out = self.stem1(x)
        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)
        branch1 = self.pool(out)

        out = Concatenate()([branch1, branch1])
        out = self.stem3(out)

        return out

class BasicConv2D(Model):
    def __init__(self, out_channels, activation=True, **kwargs):
        super(BasicConv2D, self).__init__()
        self.conv = Conv2D(filters=out_channels, use_bias=False, **kwargs)
        self.norm = BatchNormalization()
        self.activation = activation
    
    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            return Activation('relu')(x)
        else:
            return x

class PeleeNet(Model):
    def __init__(self, growth_rate=32, block_config=[3,4,8,6], num_init_features=32,
                 bottleneck_width=[1,2,4,4], drop_rate=0.5, num_classes=100):

        super(PeleeNet, self).__init__()

        self.features = Sequential(
            _StemBlock(num_init_features))

        if type(growth_rate) is list:
            growth_rates = growth_rate
            assert len(growth_rates) == 4
        else:
            growth_rates = [growth_rate] * 4

        if type(bottleneck_width) is list:
            bottleneck_widths = bottleneck_width
            assert len(bottleneck_widths) == 4
        else:
            bottleneck_widths = [bottleneck_width] * 4

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bottleneck_widths[i], growth_rate=growth_rates[i], drop_rate=drop_rate)
            self.features.add(block)
            num_features = num_features + num_layers * growth_rates[i]

            self.features.add(BasicConv2D(num_features, kernel_size=1, strides=1))

            if i != len(block_config) - 1:
                self.features.add(AveragePooling2D(2))
                num_features = num_features

        # Dense layer
        self.classifier = Dense(num_classes)
        self.drop_rate = drop_rate

    def call(self, x):
        features = self.features(x)
        out = AveragePooling2D(2)(features)

        if self.drop_rate > 0:
            out = Dropout(self.drop_rate)(out)
        out = self.classifier(out)
        return out

def main():
    parser = argparse.ArgumentParser(description='PeleeNet Trainer')
    parser.add_argument('--input_dir', help="Directory containing training data (eg. /workspace/data)")
    parser.add_argument('--output_dir', help="Directory to save model to disk (eg. /tmp/model_dir)")
    parser.add_argument('--epochs', help="Number of training epochs")
    parser.add_argument('--model_name', help="Name of the model being trained")
    parser.add_argument('--model_version', help="Version of the model (eg. 1.0.0 (versioning scheme independent))")
    parser.add_argument('--data_augment', help="Enable or disable data augmentation")
    parser.add_argument('--resize', help="Resize training data (eg. 32 (where original image size is (224,224) this would resize the image to (256, 256)))")
    parser.add_argument('--crop_pct', help="Percentage to center crop training image (eg. 0.5 will center crop to the middle 50% of pixels in the image)")
    parser.add_argument('--subtract_pixel_mean', help="Enable or disable subtracting the pixel mean from input image batches")
    parser.add_argument('--batch_size', help="Batch size for batching training data (eg. 128)")
    parser.add_argument('--learning_rate', help="Learning rate to use with the optimizer we choose on our model (eg. 1e-3 or 0.003)")
    parser.add_argument('--dataset_split', nargs='+', help="What splits to use for partitioning data between training, validation, and test (eg. (0.7, 0.15, 0.15) (repsectively))")
    parser.add_argument('--growth_rate', help="Growth Rate as defined in the PeleeNet paper (eg. 32)")
    parser.add_argument('--bottle_neck_width', help="Bottle Neck Width as defined in the PeleeNet paper (eg. 1)")
    args = parser.parse_args()

    EPOCHS = int(args.epochs)
    LEARNING_RATE = float(args.learning_rate)
    DATA_AUGMENTATION = args.data_augment
    RESIZE = int(args.resize)
    CROP_PERCENT = float(args.crop_pct)
    GROWTH_RATE = int(args.growth_rate)
    BOTTLENECK_WIDTH = int(args.bottle_neck_width)

    # TODO For data management, is there a way we can automate this process
    # for users whom use our platform(s)? Something to investigate when 
    # looking into what data management means? API calls to FS served by
    # Dell EMC storage array?
    MODEL_DIRECTORY = os.path.join(args.output_dir, args.model_name)
    MODEL_VERSION = args.model_version

    if args.batch_size:
        BATCH_SIZE = int(args.batch_size)
    else:
        BATCH_SIZE = 128
    
    dataset_splits = args.dataset_split

    TRAIN_SIZE = dataset_splits[0]
    VALIDATION_SIZE = dataset_splits[1]
    TEST_SIZE = dataset_splits[2]

    # load data

    (cifar100_train, cifar100_test), cifar100_info = tfds.load(
                                                              "cifar100",
                                                              split=["train", "test"],
                                                              shuffle_files=False,
                                                              as_supervised=True,
                                                              with_info=True
                                                              )

    def normalize(image: tf.data.Dataset, label: tf.data.Dataset) -> tf.data.Dataset:
        """ Normalize the pixel data within the images
        
        Arguments:
            image {tf.data.Dataset} -- Images contained within the TF Dataset
            label {tf.data.Dataset} -- Labels contained within the TF Dataset
        
        Returns:
            tf.dataset.Dataset -- Normalized images as TF Dataset
        """
        return tf.cast(image, tf.float32) / 255.0, label
    
    def augment(image: tf.data.Dataset, label: tf.data.Dataset) -> tf.data.Dataset:
        """ Augment the datasets
        
        Arguments:
            image {tf.data.Dataset} -- Images contained within the TF Dataset
            label {tf.data.Dataset} -- Labels contained within the TF Dataset
        
        Returns:
            tf.dataset.Dataset -- Augmented Images
        """
        original_img = image
        aug_img = tf.image.resize(original_img, ((224+RESIZE), (224+RESIZE)))
        aug_img = tf.image.central_crop(aug_img, central_fraction=CROP_PERCENT)
        return aug_img, label

    cifar100_train = cifar100_train.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    cifar100_train = cifar100_train.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    cifar100_train = cifar100_train.cache()
    # shuffle our dataset respective of the number of training examples
    cifar100_train = cifar100_train.shuffle(cifar100_info.splits['train'].num_examples)
    cifar100_train = cifar100_train.batch(BATCH_SIZE)
    cifar100_train = cifar100_train.prefetch(tf.data.experimental.AUTOTUNE)

    INPUT_SIZE = cifar100_info.features['image'].shape
    NUM_CLASSES = cifar100_info.features['label']

    print(f"Model output directory : {args.output_dir}/{args.model_name}")

    loss_object = tf.losses.SparseCategoricalCrossentropy()
    optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    validation_loss = tf.keras.metrics.Mean(name='validation_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    model = PeleeNet()

    @tf.function
    def train_step(images: tf.data.Dataset.batch, labels: tf.data.Dataset.batch):
        """Training step for our model

        Arguments:
            images {tf.data.Dataset.batch} -- Single batch of images for model training
            labels {tf.data.Dataset.batch} -- Single batch of labels for model training
        """
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)

    @tf.function
    def validation_step(images: tf.data.Dataset.batch, labels: tf.data.Dataset.batch):
        """[summary]

        Arguments:
            images {tf.data.Dataset.batch} -- [description]
            labels {tf.data.Dataset.batch} -- [description]
        """
        predictions = model(images)
        v_loss = loss_object(labels, predictions)

        validation_loss(v_loss)

    @tf.function
    def test_step(images: tf.data.Dataset.batch, labels: tf.data.Dataset.batch):
        """[summary]

        Arguments:
            images {tf.data.Dataset.batch} -- [description]
            tf {[type]} -- [description]
        """
        predictions = model(images)
        t_loss = loss_object(predictions, labels)

        test_loss(t_loss)


    #TODO clean up this logic for directory creation 
    if os.path.isdir(MODEL_DIRECTORY):
        shutil.rmtree(MODEL_DIRECTORY)
    os.mkdir(MODEL_DIRECTORY)
    os.mkdir(os.path.join(MODEL_DIRECTORY, MODEL_VERSION))

    checkpoint_dir = os.path.join(MODEL_DIRECTORY, str(MODEL_VERSION), 'ckpt')
    tensorboard_dir = os.path.join(MODEL_DIRECTORY, str(MODEL_VERSION), 'logs')
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        shutil.rmtree(tensorboard_dir)
    os.mkdir(checkpoint_dir)
    os.mkdir(tensorboard_dir)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_path = checkpoint_dir + '/'
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_path, max_to_keep=5)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = tensorboard_dir + '/gradient_tape/' + current_time + '/train'
    test_log_dir = tensorboard_dir + '/gradient_tape/' + current_time + '/test'
    print(f"Training Log Directory : {train_log_dir}")
    print(f"Testing Log Directory : {test_log_dir}")
    validation_log_dir = tensorboard_dir + '/gradient_tape/' + current_time + '/validation'
    os.makedirs(train_log_dir)
    os.makedirs(test_log_dir)
    os.makedirs(validation_log_dir)

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    validation_summary_writer =tf.summary.create_file_writer(validation_log_dir)

    for epoch in range(EPOCHS):
        print(f"Starting epoch number {(epoch + 1)}...")
        for step, (images, labels) in enumerate(cifar100_train):
            train_step(images, labels)
        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
        
        # print(f"Testing epoch number {(epoch + 1)}...")
        # test_step(cifar100_test['images'], cifar100_test['labels'])
        # with test_summary_writer.as_default():
        #     tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
        print(f"Epoch : {epoch+1}, Training Loss : {train_loss.result()}")
        checkpoint_manager.save(checkpoint_number=None)
    
        train_loss.reset_states()
        validation_loss.reset_states()
        test_loss.reset_states()
    
    model.save((MODEL_DIRECTORY + "/" + "PELEENET" + "-" + str(MODEL_VERSION) + '-' + str(EPOCHS)))

    tensorboard_metadata = {
        "outputs": [{
            "type": "tensorboard",
            "source": f"'{(tensorboard_dir + '/gradient_tape')}'",
        }]
    }

    # with open('/mlpipeline-ui-metadata.json', 'w') as f:
    #     json.dump(tensorboard_metadata, f)

    # with open('/output.txt', 'w') as f:
    #     f.write(args.output_dir)

if __name__ == '__main__':
    main()
