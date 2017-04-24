
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


## Usage:

Download the project and run command(Please download weights file from http://pan.baidu.com/s/1jH8lBee to ./model.):

```python
python mosaic_porngraphic_cnn_demo.py the_path_of_image

```
the_path_of_image is the input image to be detected.


Output is The PORNGRAPHIC probability.

## Example:

Run:
```python
python mosaic_porngraphic_cnn_demo.py ./data/test/6.jpg

```

Output:
```python
Mosaic: detect porngraghic
=====================================

loading weights..........
load OK!

Try to Predict input images:
=====================================
Input image is: ./data/test/6.jpg
The PORNGRAPHIC probability is: 100.000% 
```

Also, the model gives JSON Output:
```
following are return values:
{"results_output": {"porngraphic": "0.274794"}, "image_file_output": "null", "results_file_output": "null"}
```
------------------


## Models:

Mosaic Model based on Xception in keras:

- Input: [100,100,3]
- Entry Flow: [5,5,728]
- Middle Flow(repeated 8 times): [5,5,728]
- Exit Flow: [3,3,2048]
- Output FC: 1

The accuracy in test datasets is up to 96%.





