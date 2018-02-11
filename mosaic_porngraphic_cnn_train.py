#coding=UTF-8
'''
This is a demo that how to use Mosaic Model to detect image.
Usage:  python mosaic_porngraphic_cnn_demo.py the_path_of_image
Output: The PORNGRAPHIC probability

Author: boozyguo
mail: 44167841@qq.com
update: 2017-04-24

changlist:
2018-02-11: add training code

'''



from __future__ import absolute_import
from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import argparse
from keras.layers.core import Dense, Flatten
from keras.layers import Input
from keras.models import Model
from keras import optimizers
import keras as ks


print ('\r\nTraining Mosaic Model:')
print ('=====================================')

parser = argparse.ArgumentParser(description='Detect Porngraphic Images by Deep Learning with Keras.')
parser.add_argument('train_image_path', metavar='base', type=str,
                    help='Path to the training image.')
parser.add_argument('val_image_path', metavar='base', type=str,
                    help='Path to the validation image.')

args = parser.parse_args()
train_image_path = args.train_image_path
val_image_path = args.val_image_path

# parameters
batch_size = 64
nb_classes = 1
nb_epoch = 300
data_augmentation = True
lr = 0.01

# input image dimensions
img_rows, img_cols = 128, 128
img_channels = 3


# define model
img_input = Input(shape=(img_rows, img_cols,3))
xinception = ks.applications.Xception(include_top=False,weights=None,input_tensor=img_input)
output = xinception.output
output = Flatten(name='flatten')(output)
output = Dense(nb_classes, activation='sigmoid', name='predictions')(output)
model = Model(xinception.input, output)
model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.Adam(lr=lr),
                      metrics=["accuracy"])


#checkpoint callback
checkpoint = ModelCheckpoint('mosaic_porngraphic_cnn.best.hdf5', monitor='val_acc', save_best_only=True,mode='max')


# training augmentation
train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# validation augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

# train data generator
train_generator = train_datagen.flow_from_directory(
        train_image_path,  # this is the target directory
        target_size=(img_rows, img_cols),  # all images will be resized 
        batch_size=batch_size,
        class_mode='binary')  #  binary labels

# validation data generator
validation_generator = test_datagen.flow_from_directory(
        val_image_path,  # this is the validation directory
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='binary')


history = model.fit_generator(
                train_generator,
                samples_per_epoch=4096
                nb_epoch=nb_epoch,
                validation_data=validation_generator,
                callbacks=[checkpoint],
                nb_val_samples=1024)
model.save_weights('mosaic_porngraphic_cnn.epoch.hdf5')  #  save weights after training


exit()

