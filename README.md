
# Mosaic: Deep Learning Model for Porngraphic Detection


## You have just found Mosaic Models.

Mosaic is a high-level neural networks model, written in Python and capable of running on top of [keras](https://github.com/fchollet/keras) with [Tensorflow](https://github.com/tensorflow/tensorflow) backend . 

Now, It was developed with a focus on porngraphic detection. 

Use Mosaic Models if you need a deep learning Model that:

- Allows for easy and fast detecting porngraphic.
- Design and Fine-tune yourself Network (Mosaic-based).
- Runs seamlessly on CPU and GPU.

Mosaic is compatible with: __Keras 2.0.3__   __Tensorflow 1.1.0__ and __Python 2.7__.


------------------


## Guiding principles

- __Easy Usage.__ Just load the model and weights to predict a image.

- __Work with Python__. 


------------------


## Inference Usage:

Download the project and run command(Please download weights file from http://pan.baidu.com/s/1i4POGXB to ./model.):

```python
python mosaic_porngraphic_cnn_demo.py the_path_of_image

```
the_path_of_image is the input image to be detected.


Output is The PORNGRAPHIC probability.


## Training Usage:

Prepare training and validtaion images at first:

```python
python mosaic_porngraphic_cnn_train.py the_path_of_training_image the_path_of_validation_image 

```
the_path_of_training_image is the full path of training image, which should be include two class sub-directories.
the_path_of_validation_image is the full path of validation image, which should be include two class sub-directories.


The best model and last model will be saved.


------------------

## Example:

Run:
```python
python mosaic_porngraphic_cnn_demo.py ./data/test/22.jpg

```

Output:
```python
Mosaic: detect porngraghic
=====================================

loading weights..........
load OK!

Try to Predict input images:
=====================================
Input image is: ./data/test/22.jpg
The PORNGRAPHIC probability is: 0.000% 
```

Also, the model gives JSON Output:
```
following are return values:
{"results_output": {"porngraphic": "2.15399e-08"}, "image_file_output": "null", "results_file_output": "null"}
```
------------------


## Models:

Mosaic Model based on Xception in keras:

- Input: [128,128,3]
- Entry Flow: [8,8,728]
- Middle Flow(repeated 8 times): [8,8,728]
- Exit Flow: [4,4,2048]
- Output FC: 1

The accuracy in test datasets is up to 96%.


------------------


## Donate:

If the project could help you, please give us some donations. Donations will be used to fund expenses related to development (e.g. to cover equipment and server maintenance costs), to sponsor bug fixing, feature development.


![alt text](https://github.com/yongxuUSTC/deep_learning_based_speech_enhancement_keras_python/blob/master/mixture2clean_dnn/appendix/enhanced_log_sp.png)





