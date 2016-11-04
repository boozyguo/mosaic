'''
This is a demo that how to use Mosaic Model to detect image.
Usage:  python mosaic_porngraphic_cnn_demo.py the_path_of_image
Output: The PORNGRAPHIC probability

Author: boozyguo
mail: 44167841@qq.com
update: 2016-11-04
'''

# model reconstruction from JSON:
from keras.models import model_from_json
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import h5py
import os
import argparse

print ('\r\nMosaic: detect porngraghic')
print ('=====================================')

parser = argparse.ArgumentParser(description='Detect Porngraphic Images by Deep Learning with Keras.')
parser.add_argument('input_image_path', metavar='base', type=str,
                    help='Path to the image to detect.')

args = parser.parse_args()
input_image_path = args.input_image_path


# input image dimensions
img_rows, img_cols = 32, 32
img_channels = 3

# load model
if os.path.exists('./model/mosaic_porngraphic_cnn_architecture_release.json'):
    model = model_from_json(open('./model/mosaic_porngraphic_cnn_architecture_release.json').read())
    print ('loading CNN model..........')
    print ('load OK!')
else:
    print ('Not Find Model........')
    exit()

# load weights
if os.path.exists('./model/mosaic_porngraphic_cnn.release.hdf5'):
    model.load_weights('./model/mosaic_porngraphic_cnn.release.hdf5')
    print ('loading weights..........')
    print ('load OK!')
else:
    print ('Not Find Weights........')
    exit()

# try to predicts some images
print ('\r\nTry to Predict input images:')
print ('=====================================')
re_img = load_img(input_image_path)  # this is a jpg image
re_img = re_img.resize((img_cols,img_rows))
x_img = img_to_array(re_img)  # this is a Numpy array with shape (3, 32, 32)
x_img = x_img.reshape((1,) + x_img.shape)  # this is a Numpy array with shape (1, 3, 32, 32)
print ('Input image is: %s' %input_image_path)
print ('The PORNGRAPHIC probability is: %3.3f%% \r\n' %(100*model.predict_on_batch(x_img)))




exit()

