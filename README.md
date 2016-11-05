
# Mosaic: Deep Learning Model for Porngraphic Detection


## You have just found Mosaic Models.

Mosaic is a high-level neural networks model, written in Python and capable of running on top of [keras](https://github.com/fchollet/keras). 

Now, It was developed with a focus on porngraphic detection. 

Use Mosaic Models if you need a deep learning Model that:

- Allows for easy and fast detecting porngraphic.
- Design and Fine-tune yourself Network (Mosaic-based).
- Runs seamlessly on CPU and GPU.

Mosaic is compatible with: __Keras 1.1.1__ and __Python 2.7__.


------------------


## Guiding principles

- __Easy Usage.__ Just load the model and weights to predict a image.

- __Work with Python__. 


------------------


## Usage:

Download the project and run command:

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
loading CNN model..........
load OK!
loading weights..........
load OK!

Try to Predict input images:
=====================================
Input image is: ./data/test/6.jpg
The PORNGRAPHIC probability is: 100.000% 
```

------------------


## Models:

Mosaic Model contains 2 CNN layers and 3 FC layers:

- Input: [3,32,32]
- CNN layer1: [64,3,3]
- CNN layer2: [64,3,3]
- FC layer1: 128
- FC layer2: 128
- Output FC layer3: 1

The accuracy in test datasets is up to 96%.





