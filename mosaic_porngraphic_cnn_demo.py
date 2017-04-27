#coding=UTF-8
'''
This is a demo that how to use Mosaic Model to detect image.
Usage:  python mosaic_porngraphic_cnn_demo.py the_path_of_image
Output: The PORNGRAPHIC probability

Author: boozyguo
mail: 44167841@qq.com
update: 2017-04-24

changlist:
2016-11-04: first version, using 2 CNN layers and 3 FC layers
2017-04-24: second version, using Xception
2017-04-27: second version, change input size to 128
'''

# model reconstruction from JSON:

from keras.preprocessing.image import img_to_array, load_img

import h5py
import os
import argparse
import json
from keras.layers.core import Dense, Flatten
from keras.layers import Input
from keras.models import Model
from keras import optimizers
import keras as ks


print ('\r\nMosaic: detect porngraghic')
print ('=====================================')

parser = argparse.ArgumentParser(description='Detect Porngraphic Images by Deep Learning with Keras.')
parser.add_argument('input_image_path', metavar='base', type=str,
                    help='Path to the image to detect.')

args = parser.parse_args()
input_image_path = args.input_image_path


# input image dimensions
img_rows, img_cols = 128, 128
img_channels = 3
n_classes = 1

# define model
img_input = Input(shape=(img_rows, img_cols,3))
xinception = ks.applications.Xception(include_top=False,weights=None,input_tensor=img_input)
output = xinception.output
output = Flatten(name='flatten')(output)
output = Dense(n_classes, activation='sigmoid', name='predictions')(output)
model = Model(xinception.input, output)
model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.Adam(),
                      metrics=["accuracy"])


# load weights
if os.path.exists('./model/0.12-loss_18epoch_128x128_aug_0.001lr_run0_Xception_128_1493071505.11time'):
    model.load_weights('./model/0.12-loss_18epoch_128x128_aug_0.001lr_run0_Xception_128_1493071505.11time')
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
x_img = img_to_array(re_img)  # this is a Numpy array with shape (100,100,3)
x_img /= 255
x_img -= 0.5
x_img *= 2
x_img = x_img.reshape((1,) + x_img.shape)  # this is a Numpy array with shape (1, 100, 100, 3)
print ('Input image is: %s' %input_image_path)
s = model.predict_on_batch(x_img)
#print ('The PORNGRAPHIC probability is: %3.3f%% \r\n' %(100*model.predict_on_batch(x_img)))
print ('The PORNGRAPHIC probability is: %3.3f%% \r\n' %(100*s))
# exit()
#print (s[0])
print ('following are return values:' )

detectvalue={}
detectvalue["porngraphic"] = str(s[0,0])

returnvalue={}
returnvalue["results_output"] = detectvalue
returnvalue["image_file_output"] = "null"
returnvalue["results_file_output"] = "null"

print json.dumps(returnvalue)

exit()
# return (100*model.predict_on_batch(x_img)
