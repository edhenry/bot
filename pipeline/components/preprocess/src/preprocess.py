import argparse
import json
import os
from os import listdir
from os.path import isfile, join

import cv2
import IPython
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.python.lib.io import file_io

tf.compat.v1.enable_eager_execution()

def main():
    parser = argparse.ArgumentParser(description='Dataset Preprocessor')
    parser.add_argument('--input_dir', help='Preprocessed Root Data Directory')
    parser.add_argument('--output_dir', help='Postprocessed Root Data Directory')

    args = parser.parse_args()
    
    # Create a dictionary describing the features.
    # TODO : define in external protobuf defn
    image_feature_description = {
        'timestamp': tf.FixedLenFeature([], tf.float32),
        'image': tf.FixedLenFeature([], tf.string),
        'steering_theta': tf.FixedLenFeature([], tf.float32),
        'accelerator': tf.FixedLenFeature([], tf.float32),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'capture_height': tf.FixedLenFeature([], tf.int64),
        'capture_width': tf.FixedLenFeature([], tf.int64),
        'capture_fps': tf.FixedLenFeature([], tf.int64),
        'num_channels': tf.FixedLenFeature([], tf.int64)
    }

    def read_records(directory: str) -> list:
        """
        Read all files in a directory and return a list containing all 
        of the filenames.
        """
        filenames = [f for f in listdir(directory) if isfile(join(directory, f))]
        return filenames

    def create_tf_datasets(filenames: list) -> list:
        raw_image_datasets = []

        for record in read_records(args.input_dir):
            raw_image_datasets.append(tf.data.TFRecordDataset(f'{args.input_dir}/{record}'))

        return raw_image_datasets

    # raw_image_datasets = []

    # for record in tfrecords:
    #     raw_image_datasets.append(tf.data.TFRecordDataset(f'tfrecords/tfrecords/{record}'))

    def parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
        return tf.parse_single_example(example_proto, image_feature_description)

    def parse_datasets(datasets: list) -> list:
        parsed_datasets = []

        for raw_image_dataset in datasets:
            parsed_datasets.append(raw_image_dataset.map(parse_image_function))

        return parsed_datasets

    def create_videos(datasets):
        """
        Create videos to view the output of our collected dataset.
        """

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (100,175)
        fontScale              = .35
        fontColor              = (255, 255, 255)
        lineType               = 1

        # quick hack to create videos from TFRecords
        video_count = 0

        for dataset in datasets:
            try:
                video_out = cv2.VideoWriter(f'{args.input_dir}/videos/segment-{video_count}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (224,224))
                i = 0
                for image_features in dataset:
                    raw_img = image_features['image'].numpy()
                    array_str = np.frombuffer(raw_img, np.uint8).reshape(224, 224, 3)
                    _, ret = cv2.imencode('.jpg', array_str)

                    cv2.imwrite(img=array_str, filename=f'{args.input_dir}/videos/{i}.jpg')
                    text = f"Steering : {image_features['steering_theta']} \n Accelerator : {image_features['accelerator']}"

                    img = cv2.imread(filename=f'{args.input_dir}/videos/{i}.jpg')
                    y0, dy = 175, 20
                    for l, line in enumerate(text.split('\n')):
                        y = y0 + l*dy
                        cv2.putText(img, line, (0, y), font, fontScale, fontColor, lineType)

                    video_out.write(img)
                    os.remove(f'{args.input_dir}/videos/{i}.jpg')
                    i += 1
                video_out.release()
                video_count += 1
            except:
                print(f"Corrupt dataset encountered! Skipping...")


    records = read_records(args.input_dir)
    tfrecords = create_tf_datasets(records)
    parsed_datasets = parse_datasets(tfrecords)
    create_videos(parsed_datasets)

    print(f'input_dir: {args.input_dir}')
    print(f'output_dir: {args.output_dir}')


if __name__ == '__main__':
    main()
