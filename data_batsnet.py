import numpy as np
import os
import warnings
import tensorflow as tf
#tf.enable_eager_execution()
from pathlib import Path
import tensorflow_datasets as tfds

ip_dir = '/content/our_data'


# step 3: parse every image in the dataset using map
def mnist_x(x_orig_name):
    x_orig_name=ip_dir+'/batsnet_train/1/'+ x_orig_name

    image_string = tf.io.read_file(x_orig_name)


    image_decoded = tf.image.decode_png(image_string, channels=1)  #change channels back to 3
    x_orig = tf.cast(image_decoded, dtype=tf.float32) / image_decoded.dtype.max
  
    x=tf.image.resize(x_orig, [24,24])

    return x


def mnist_gx(x_orig_name):

    # if not training, return a constant value--it will unused but needs to be same shape to avoid TensorFlow errors
    x_augmented_name =ip_dir+'/augmented/'+ x_orig_name

    image_string = tf.io.read_file(x_augmented_name)
    image_decoded = tf.image.decode_png(image_string, channels=1)  #change channels back to 3

    x_augmented = tf.cast(image_decoded, dtype=tf.float32) / image_decoded.dtype.max

  
    gx=tf.image.resize(x_augmented, [24,24])

    return gx


def pre_process_data(ds, is_training, **kwargs):
    """
    :param ds: TensorFlow Dataset object
    :param info: TensorFlow DatasetInfo object
    :param is_training: indicator to pre-processing function
    :return: the passed in data set with map pre-processing applied
    """
    # apply pre-processing function for given data set and run-time conditions
    return ds.map(lambda d: {'x': mnist_x(d),
                            'gx': mnist_gx(d),
                            'label': 1,
                            'Name': d},
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

#    return ds.map(mnist_x)
        # raise Exception('Unsupported data set: ' + info.name)


def configure_data_set(ds, batch_size, is_training, **kwargs):
    """
    :param ds: TensorFlow data set object
    :param info: TensorFlow DatasetInfo object   -> REMOVED
    :param batch_size: batch size
    :param is_training: indicator to pre-processing function
    :return: a configured TensorFlow data set object
    """
    # enable shuffling and repeats
    #disabling the shuffle
    ds = ds.shuffle(10 * batch_size, reshuffle_each_iteration=True).repeat(1)

    # batch the data before pre-processing
 
    # pre-process the data set
    with tf.device('/cpu:0'):
        ds = pre_process_data(ds, is_training, **kwargs)

    ds = ds.batch(batch_size)

    # enable prefetch
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def load(data_set_name, **kwargs):
    """
    :param data_set_name: data set name--call tfds.list_builders() for options
    :return:
        train_ds: TensorFlow Dataset object for the training data
        test_ds: TensorFlow Dataset object for the testing data
        info: data set info object
    """
    #ideas from https://stackoverflow.com/questions/37340129/tensorflow-training-on-my-own-image
    #step-1 Get list of filenames      
    files_list = []
    for path in (Path(ip_dir+ '/batsnet_train/1').rglob('*.png')):
        files_list.append(str(path).split('/')[-1])

    #Step-2 create a dataset returning slices of `filenames`
    ds = tf.data.Dataset.from_tensor_slices( tf.constant(files_list))
    # configure the data sets
    train_ds = configure_data_set(ds=ds, is_training=True, **kwargs)

    ##Making the same dataset again to get predictions
    ds2 = tf.data.Dataset.from_tensor_slices( tf.constant(files_list))
    test_ds = configure_data_set(ds=ds2, is_training=True, **kwargs)  #is_training doesn't matter becuase we are not using for validation



    return train_ds, test_ds
