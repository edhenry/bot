from __future__ import division, absolute_import, print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf

import tensorflow_datasets as tfds
import tensorflow_addons on tfa

tfds.disable_progress_bat()

def load_dataset(dataset: str, split: str, shuffle: bool, with_info: bool) -> tf.dataset.Dataset:
    """Load a TensorFlow dataset for use at later steps in the pipeline
    
    Arguments:
        dataset {str} -- name of the built-in dataset
    
    Returns:
        tf.dataset.Dataset -- TensorFlow Dataset
    """

    dataset = tfds.load(name=dataset, split=split, shuffle_files=shuffle, with_info=with_info)
    return dataset

def augment_dataset(images: tf.dataset.Dataset) -> tf.dataset.Dataset:
    """Method to use for augmenting our dataset
    
    Arguments:
        augment {bool} -- Bool to enable or disable data augmentation
    
    Returns:
        tf.dataset.Dataset -- tensorflow dataset
    """
    


