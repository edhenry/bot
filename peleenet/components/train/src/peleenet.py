import argparse
import datetime
import json
import math
import os
import pickle
import shutil
from collections import OrderedDict
from random import randrange
from typing import List, Tuple

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import tensorflow_addons as tfa  # type: ignore
from PIL import Image  # type: ignore
from tensorflow.keras import Sequential, regularizers  # type: ignore
from tensorflow.keras.callbacks import (ModelCheckpoint,  # type: ignore
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import (Activation,  # type: ignore
                                     AveragePooling2D, BatchNormalization,
                                     Concatenate, Conv2D, Dense, Dropout,
                                     Flatten, GlobalAveragePooling2D, Input,
                                     MaxPool2D)
from tensorflow.keras.models import Model  # type: ignore

import tensorflow_datasets as tfds  # type: ignore


class _DenseLayer(Model):
    def __init__(self, num_input_features, growth_rate, bottleneck_width, drop_rate):
        super(_DenseLayer, self).__init__()

        growth_rate: int = int(growth_rate / 2)
        inter_channel: int = int(growth_rate * bottleneck_width / 4) * 4

        if inter_channel > num_input_features / 2:
            inter_channel = int(num_input_features / 8) * 4
            print(f'adjusting inter_channel to {inter_channel}')
        
        self.branch1a = BasicConv2D(inter_channel, kernel_size=1, padding="same")
        self.branch1b = BasicConv2D(growth_rate, kernel_size=3, padding="same")

        self.branch2a = BasicConv2D(inter_channel, kernel_size=1, padding="same")
        self.branch2b = BasicConv2D(growth_rate, kernel_size=3, padding="same")
        self.branch2c = BasicConv2D(growth_rate, kernel_size=3, padding="same")

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
        self.conv = Conv2D(filters=out_channels, use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(5e-4), **kwargs)
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
                 bottleneck_width=[1,2,4,4], drop_rate=0.5, num_classes=1000):

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
        self.classifier = Dense(num_classes, kernel_initializer='glorot_uniform')
        self.drop_rate = drop_rate

    def call(self, x):
        features = self.features(x)
        out = GlobalAveragePooling2D()(features)

        if self.drop_rate > 0:
            out = Dropout(self.drop_rate)(out)
        out = self.classifier(out)
        return out

class ImageAugmentation:
    """
    Resize all images in dataset to (224,224,3) as there are variable sized images in some datasets
    """
    def __init__(self):
        pass

    def __call__(self, image, label):
        aug_img = tf.image.resize(image, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return aug_img, label

class TrainingImageAugmentation:
    def __init__(self, log_dir: str, max_images: int, name: str,
                 input_size: int, scale_img: int, resize: int,
                 batch_size: int):
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.max_images: int = max_images
        self.name: str = name
        self.input_size: int = input_size
        self.resize: int = resize
        self.batch_size: int = batch_size
        self.scale_img: int = scale_img

        self._counter: int = 0

    def __call__(self, image, label):
        image = tf.cast(image, tf.float32) / 255.0
        #aug_img = tf.image.per_image_standardization(image)
        aug_img = tf.image.resize(image, (((self.input_size * self.scale_img) + self.resize), ((self.input_size * self.scale_img) + self.resize)), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        #aug_img = tf.image.random_crop(aug_img, size=(224, 224, 3))
        aug_img = tf.image.random_flip_left_right(aug_img)
        aug_img = tf.image.resize(aug_img, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        with self.file_writer.as_default():
            tf.summary.image(
                self.name,
                aug_img,
                step=self._counter,
                max_outputs = self.max_images
            )
        
        self._counter += 1
        return aug_img, label

class TestingImageAugmentation:
    def __init__(self, log_dir: str, max_images: int, name: str,
                 input_size: int, scale_img: int, resize: int,
                 batch_size: int) -> None:
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.max_images: int = max_images
        self.name: str = name
        self.input_size: int = input_size
        self.resize: int = resize
        self.batch_size: int = batch_size
        self.scale_img: int = scale_img

        self._counter: int = 0

    def __call__(self, image: tf.data.Dataset, label: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        img: tf.data.Dataset = tf.cast(image, tf.float32) / 255.0
        #aug_img = tf.image.per_image_standardization(image)
        aug_img: tf.data.Dataset = tf.image.resize(img, (((self.input_size * self.scale_img) + self.resize), ((self.input_size * self.scale_img) + self.resize)), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        aug_img = tf.image.resize(aug_img, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        with self.file_writer.as_default():
            tf.summary.image(
                self.name,
                aug_img,
                step=self._counter,
                max_outputs = self.max_images
            )
        
        self._counter += 1
        return aug_img, label

def main():
    parser = argparse.ArgumentParser(description='PeleeNet Trainer')
    parser.add_argument('--input_dir', help="Directory containing training data (eg. /workspace/data)")
    parser.add_argument('--output_dir', help="Directory to save model to disk (eg. /tmp/model_dir)")
    parser.add_argument('--epochs', help="Number of training epochs")
    parser.add_argument('--model_name', help="Name of the model being trained")
    parser.add_argument('--model_version', help="Version of the model (eg. 1.0.0 (versioning scheme independent))")
    parser.add_argument('--data_augment', help="Enable or disable data augmentation")
    parser.add_argument('--resize', help="Resize training data (eg. 32 (where original image size is (224,224) this would resize the image to (256, 256)))")
    parser.add_argument('--scale_img', help="Factor by which to scale the input image (eg. 7 (if the input image is 32x32x3 (HWC) the output would be (224,224,3)))")
    parser.add_argument('--crop_pct', help="Percentage to center crop training image (eg. 0.5 will center crop to the middle 50% of pixels in the image)")
    parser.add_argument('--subtract_pixel_mean', help="Enable or disable subtracting the pixel mean from input image batches")
    parser.add_argument('--batch_size', help="Batch size for batching training data (eg. 128)")
    parser.add_argument('--learning_rate', help="Learning rate to use with the optimizer we choose on our model (eg. 1e-3 or 0.003)")
    parser.add_argument('--momentum', help="Momentum to use for the SGD Optimizer")
    parser.add_argument('--lr_patience', help='Number of epochs with no improvement after which learning rate will be reduced. (eg. 5)')
    parser.add_argument('--dropout', help="Percentage of dropout to add to the network (eg .5 == 50% dropout rate")
    parser.add_argument('--dataset_split', nargs='+', type=float, help="What splits to use for partitioning data between training, validation, and test (eg. 0.7 0.15 0.15) (repsectively))")
    parser.add_argument('--growth_rate', help="Growth Rate as defined in the PeleeNet paper (eg. 32)")
    parser.add_argument('--bottle_neck_width', nargs="+", type=int, help="Bottle Neck Width as defined in the PeleeNet paper (eg. 1 2 4 4)")
    parser.add_argument('--num_classes', help="Number of classes contained within a dataset. (eg. 1000 for ImageNet)")
    parser.add_argument('--input_size', help="Input size of the dataset (eg. 224 for images with (224,224,3) dimensions)")
    parser.add_argument('--prefetch_size', help="Number of batches to prefetch for model training (eg. 5)")
    parser.add_argument('--shuffle_buffer', help="Number of data points to add to shuffle buffer (eg. 10000)")
    args = parser.parse_args()

    EPOCHS = int(args.epochs)
    LEARNING_RATE = float(args.learning_rate)
    MOMENTUM = float(args.momentum)
    #TODO(ehenry): Make dataset augmentation optional as an argument for hyperparameter sweeps
    DATA_AUGMENTATION = args.data_augment
    RESIZE = int(args.resize)
    SCALE_IMG = int(args.scale_img)
    DROPOUT = float(args.dropout)
    PATIENCE = int(args.lr_patience)
    INPUT_DIR = str(args.input_dir)
    NUM_CLASSES = int(args.num_classes)
    INPUT_SIZE = int(args.input_size)
    PREFETCH_SIZE = int(args.prefetch_size)
    SHUFFLE_BUFFER = int(args.shuffle_buffer)
    
    #TODO(ehenry): This can likely be combined with the DATA_AUGMENTATION flag above. 
    # It should be made optional for via command line argument for hyperparameter sweeps
    if args.crop_pct:
        CROP_PERCENT = float(args.crop_pct)
    else:
        pass

    if args.growth_rate:
        GROWTH_RATE = int(args.growth_rate)
    else:
        GROWTH_RATE = 32

    if args.bottle_neck_width:
        BOTTLENECK_WIDTH = list(args.bottle_neck_width)
    else:
        BOTTLENECK_WIDTH = [1,2,4,4]

    # TODO(ehenry) For data management, is there a way we can automate this process
    # for users whom use our platform(s)? Something to investigate when 
    # looking into what data management means? API calls to FS served by
    # Dell EMC storage array?
    MODEL_DIRECTORY = os.path.join(args.output_dir, args.model_name)
    MODEL_VERSION = args.model_version

    if args.batch_size:
        BATCH_SIZE = int(args.batch_size)
    else:
        BATCH_SIZE = 128

    #TODO(ehenry) clean up this logic for directory creation 
    if os.path.isdir(MODEL_DIRECTORY):
        os.mkdir(os.path.join(MODEL_DIRECTORY, MODEL_VERSION))
    else:
        pass

    checkpoint_dir = os.path.join(MODEL_DIRECTORY, str(MODEL_VERSION), 'ckpt')
    tensorboard_dir = os.path.join(MODEL_DIRECTORY, str(MODEL_VERSION), 'logs')
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        shutil.rmtree(tensorboard_dir)
    os.mkdir(checkpoint_dir)
    os.mkdir(tensorboard_dir)

    #TODO(ehenry): Clean up this mess of logging locations on filesystem
    current_time: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir: str = tensorboard_dir + '/gradient_tape/' + current_time + '/train'
    train_img_dir: str = tensorboard_dir + '/gradient_tape/' + current_time + '/train/images'
    test_img_dir: str = tensorboard_dir + '/gradient_tape/' + current_time + '/test/images'
    test_log_dir: str = tensorboard_dir + '/gradient_tape/' + current_time + '/test'
    print(f"Training Log Directory : {train_log_dir}")
    print(f"Testing Log Directory : {test_log_dir}")
    validation_log_dir: str = tensorboard_dir + '/gradient_tape/' + current_time + '/validation'
    os.makedirs(train_log_dir)
    os.makedirs(test_log_dir)
    os.makedirs(validation_log_dir)
    
    # dataset_splits = args.dataset_split

    # TRAIN_SIZE = dataset_splits[0]
    # VALIDATION_SIZE = dataset_splits[1]
    # TEST_SIZE = dataset_splits[2]

    # load data

    #TODO(ehenry): Break out data loading functionality into separate module
    # image_net = tfds.builder("cifar10")
    # download_config = tfds.download.DownloadConfig()
    # download_config.manual_dir="/home/tom/tensorflow_datasets"
    # #download_config.extract_dir="/mnt/tensorflow_datasets"
    # download_config.compute_stats="skip"
    # image_net.download_and_prepare(download_config=download_config)

    # # image_net.as_dataset()
    # # image_net_train, image_net_valid = image_net['train'], image_net['valid']

    (train, test), info = tfds.load("imagenet2012",
                                    split=["train", "validation"],
                                    shuffle_files=True,
                                    as_supervised=True,
                                    with_info=True,
                                    data_dir=INPUT_DIR)

    #TODO(ehenry): Match learning rate scheduler to peleenet paper -- for now using peicewiseconstantdecay
    def lr_scheduler(init_lr: float, num_epochs: int, iterations_per_epoch: int, iterations: int) -> Tuple[List, List]:
        """Scheduler for use in reducing learning rate during training as outlined in the original PeleeNet Paper

        Arguments:
            init_lr {float} -- Initial learning rate
            num_epochs {int} -- Total number of training epochs
            iterations_per_epoch {int} -- Total number of iterations_per_epoch (total number of training examples)
            iterations {int} -- Total number of steps per epoch

        Returns:
            Tuple[List, List] -- List of boundaires and list of values for use in optimization object
        """
        
        # Lists of boundaries and values for use in PiecewiseConstantDecay learning rate
        boundaries = []
        values = []

        learning_rate: float = init_lr
        T_total: int = num_epochs * iterations_per_epoch
        for i in range(EPOCHS):
            for e in range(iterations):
                T_cur: int = (i % num_epochs) * iterations_per_epoch + e
                lr: float = 0.5 * learning_rate * (1 + math.cos(math.pi * T_cur / T_total))
            boundaries.append(i+1)
            values.append(lr)

        return boundaries[:-1], values

    BATCH_SIZE_AUGMENTATION = ImageAugmentation()

    TRAIN_AUGMENTATION = TrainingImageAugmentation(train_img_dir, max_images=5, name="Augmented Training Images", 
                                                   input_size=INPUT_SIZE, scale_img=SCALE_IMG, resize=RESIZE,
                                                   batch_size=BATCH_SIZE)
    
    TEST_AUGMENTATION = TestingImageAugmentation(test_img_dir, max_images=5, name="Augmented Testing Images",
                                                 input_size=INPUT_SIZE, scale_img=SCALE_IMG, resize=RESIZE,
                                                 batch_size=BATCH_SIZE)

    # Shuffle our dataset, and reshuffle after each epoch
    train = train.shuffle(SHUFFLE_BUFFER, reshuffle_each_iteration=True)

    # Resize all images, create training batches, augment the images, and prefetch batches
    train = train.map(BATCH_SIZE_AUGMENTATION, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE, drop_remainder=True).map(TRAIN_AUGMENTATION, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(PREFETCH_SIZE)

    # Resize all images, create testing batches, augment the images, and prefetch batches
    test = test.map(BATCH_SIZE_AUGMENTATION, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE, drop_remainder=True).map(TEST_AUGMENTATION, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(PREFETCH_SIZE)
    #test = test.prefetch(PREFETCH_SIZE)

    print(f"Model output directory : {args.output_dir}/{args.model_name}")

    # Loss object for use in tracking model loss on train/validation/test datasets
    loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Quick hack for using PiecewiseConstantDecay for learning rate decay
    # Ideally we'd want to implement our own LearningRateSchedule here, but
    # this should work for now...
    lr_boundaries, lr_values = lr_scheduler(init_lr=LEARNING_RATE, num_epochs=EPOCHS, iterations_per_epoch=info.splits['train'].num_examples, iterations=info.splits['train'].num_examples//BATCH_SIZE)
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=lr_boundaries, values=lr_values)

    # Optimizer (this one is pretty straight forward)
    optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)

    # Metrics for tracking train, validation, and test loss during training
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    validation_loss = tf.keras.metrics.Mean(name='validation_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # Train accuracy metric for use in model training
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # Test accuracy metric for use in model training
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # Create an instance of out model
    model = PeleeNet(bottleneck_width=BOTTLENECK_WIDTH, growth_rate=GROWTH_RATE, drop_rate=DROPOUT, num_classes=NUM_CLASSES)

    graph_writer = tf.summary.create_file_writer(train_log_dir)
    tf.summary.trace_on(graph=True, profiler=True)
    with graph_writer.as_default():
        tf.summary.trace_export(
            name='peleenet_trace',
            step=0,
            profiler_outdir=train_log_dir
        )

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

        train_accuracy(labels, predictions)
        train_loss(loss)

    # @tf.function
    # def validation_step(images: tf.data.Dataset.batch, labels: tf.data.Dataset.batch):
    #     """[summary]

    #     Arguments:
    #         images {tf.data.Dataset.batch} -- Batch of validation images
    #         labels {tf.data.Dataset.batch} -- Batch of validation labels
    #     """
    #     predictions = model(images)
    #     v_loss = loss_object(labels, predictions)

    #     validation_loss(v_loss)

    @tf.function
    def test_step(images: tf.data.Dataset.batch, labels: tf.data.Dataset.batch):
        """Test step for use with training our model

        Arguments:
            images {tf.data.Dataset.batch} -- Batch of testing images
            labels {tf.data.Dataset.batch} -- Batch of testing labels
        """
        # Get model preidctions
        predictions = model(images, training=False)

        # Get test dataset loss
        t_loss = loss_object(labels, predictions)

        # Set test dataset accuracy
        test_accuracy(labels, predictions)

        # Set test dataset loss
        test_loss(t_loss)

    # Checkpoint object for use in training pipeline
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_path = checkpoint_dir + '/'

    # Checkpoint manager for managing checkpoints during training 
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_path, max_to_keep=5)

    # model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['acc'])

    # checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, monitor='val_loss', verbose=1, 
    #                                                 mode='auto', save_best_only=True, save_freq='epoch', save_weights_only=False)
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=20, write_graph=True,
    #                                              update_freq='batch', profile_batch=2)

    # lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=PATIENCE,
    #                                                   verbose=1, mode='auto', cooldown=0, min_lr=0.0001)

    # print('Fit model...')
    # history = model.fit(train,
    #                     epochs=EPOCHS,
    #                     validation_data=test,
    #                     callbacks=[lr_plateau, checkpoint, tensorboard])

    # Create summary writers for writing values for visualization in TensorBoard
    #TODO(ehenry) Implement validation dataset logic
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    validation_summary_writer =tf.summary.create_file_writer(validation_log_dir)

    # Metrics to use with progress bar for model training and testing
    progbar_metrics = ['train_loss', 'val_loss', 'test_loss', 
                       'train_accuracy', 'val_accuracy', 'test_accuracy'
                       'learning_rate']

    
    for epoch in range(EPOCHS):
        print(f"\nStarting epoch number {(epoch + 1)}...")

        # Progress bar for tracking training and testing 
        bar = tf.keras.utils.Progbar(target=info.splits['train'].num_examples//BATCH_SIZE, unit_name="step", stateful_metrics=progbar_metrics)

        # Iterate over training dataset batches
        for step, (images, labels) in enumerate(train):

            # Evaluate BATCH_SIZE of images and take a gradient step
            train_step(images, labels)

            # Update the progress bar for CLI output
            bar.update(step, values=[('train_loss', train_loss.result()), ('train_accuracy', train_accuracy.result())])
        
        # Write per epoch training results to Tensorboard summary files
        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
            tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch)
            tf.summary.scalar('learning_rate', optimizer.learning_rate.numpy(), step=epoch)

        # Iterate over test dataset batches
        for step, (images, labels) in enumerate(test):

            # Evaluate the model in BATCH_SIZE of images
            test_step(images, labels)

        # Write per epoch training results to Tensorboard summary files    
        with test_summary_writer.as_default():
            tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
            tf.summary.scalar('test_accuracy', test_accuracy.result(), step=epoch)
        
        # Update progres bar with results of test dataset evaluation
        bar.update(info.splits['train'].num_examples//BATCH_SIZE, values=[('test_loss', test_loss.result()), ('test_accuracy', test_accuracy.result())])
        
        # Checkpoint the model to disk
        checkpoint_manager.save(checkpoint_number=None)

        # Reset metric states for each epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        validation_loss.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
    
    #TODO(ehenry): Evaluate if this is necessary with model checkpointing...
    model.save((MODEL_DIRECTORY + "/" + "PELEENET" + "-" + str(MODEL_VERSION) + '-' + str(EPOCHS)))

    #TODO(ehenry): Implement logic to write metadata files for use in Kubeflow pipelines
    # This specific example will allow for spawning a TensorBoard instance within Kubernetes
    # from the Kubeflow Pipelines UI
    tensorboard_metadata = {
        "outputs": [{
            "type": "tensorboard",
            "source": f"'{(tensorboard_dir + '/gradient_tape')}'",
        }]
    }

    #TODO(ehenry): Define logic for saving model metadata to the metadata module included with Kubeflow
    # with open('/mlpipeline-ui-metadata.json', 'w') as f:
    #     json.dump(tensorboard_metadata, f)

    # with open('/output.txt', 'w') as f:
    #     f.write(args.output_dir)

if __name__ == '__main__':
    main()
