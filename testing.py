import tensorflow as tf
from PIL import Image
from matplotlib import cm
import numpy as np
tf.enable_eager_execution()
filename="/media/rabi/Data/ThesisData/audio data analysis/audio-clustering/plots_15march_b/spectrograms_normalized/batsnet_train/1/WWNP-HQ_20190218_222404.wav at 6.692699.png"
image_string = tf.io.read_file(filename)

image_decoded = tf.image.decode_png(image_string, channels=1)  #change channels back to 3
# with open(x_orig_name, 'rb') as f:  
#         image_read = Image.open(f).convert('L') ## convert back to convert('RGB')

# rescale to [0, 1]
x_orig = tf.cast(image_decoded, dtype=tf.float32) / image_decoded.dtype.max

# get common shapes
height_width = [100,100]

x=tf.image.resize(x_orig, height_width)

#im = Image.fromarray(np.uint8(cm.gist_earth(np.reshape(x.numpy(), (100,100)))*255))


x=tf.reshape(x, (-1, 100,100,1))
