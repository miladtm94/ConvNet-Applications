# Convolutional Neural Networks: Application

Welcome to Course 4's second assignment! In this notebook, you will:

- Create a mood classifer using the TF Keras Sequential API
- Build a ConvNet to identify sign language digits using the TF Keras Functional API

**After this assignment you will be able to:**

- Build and train a ConvNet in TensorFlow for a __binary__ classification problem
- Build and train a ConvNet in TensorFlow for a __multiclass__ classification problem
- Explain different use cases for the Sequential and Functional APIs

To complete this assignment, you should already be familiar with TensorFlow. If you are not, please refer back to the **TensorFlow Tutorial** of the third week of Course 2 ("**Improving deep neural networks**").

## Important Note on Submission to the AutoGrader

Before submitting your assignment to the AutoGrader, please make sure you are not doing the following:

1. You have not added any _extra_ `print` statement(s) in the assignment.
2. You have not added any _extra_ code cell(s) in the assignment.
3. You have not changed any of the function parameters.
4. You are not using any global variables inside your graded exercises. Unless specifically instructed to do so, please refrain from it and use the local variables instead.
5. You are not changing the assignment code where it is not required, like creating _extra_ variables.

If you do any of the following, you will get something like, `Grader not found` (or similarly unexpected) error upon submitting your assignment. Before asking for help/debugging the errors in your assignment, check for these first. If this is the case, and you don't remember the changes you have made, you can get a fresh copy of the assignment by following these [instructions](https://www.coursera.org/learn/convolutional-neural-networks/supplement/DS4yP/h-ow-to-refresh-your-workspace).

## Table of Contents

- [1 - Packages](#1)
    - [1.1 - Load the Data and Split the Data into Train/Test Sets](#1-1)
- [2 - Layers in TF Keras](#2)
- [3 - The Sequential API](#3)
    - [3.1 - Create the Sequential Model](#3-1)
        - [Exercise 1 - happyModel](#ex-1)
    - [3.2 - Train and Evaluate the Model](#3-2)
- [4 - The Functional API](#4)
    - [4.1 - Load the SIGNS Dataset](#4-1)
    - [4.2 - Split the Data into Train/Test Sets](#4-2)
    - [4.3 - Forward Propagation](#4-3)
        - [Exercise 2 - convolutional_model](#ex-2)
    - [4.4 - Train the Model](#4-4)
- [5 - History Object](#5)
- [6 - Bibliography](#6)

<a name='1'></a>
## 1 - Packages

As usual, begin by loading in the packages.


```python
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator

%matplotlib inline
np.random.seed(1)
```

<a name='1-1'></a>
### 1.1 - Load the Data and Split the Data into Train/Test Sets

You'll be using the Happy House dataset for this part of the assignment, which contains images of peoples' faces. Your task will be to build a ConvNet that determines whether the people in the images are smiling or not -- because they only get to enter the house if they're smiling!  


```python
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 600
    number of test examples = 150
    X_train shape: (600, 64, 64, 3)
    Y_train shape: (600, 1)
    X_test shape: (150, 64, 64, 3)
    Y_test shape: (150, 1)


You can display the images contained in the dataset. Images are **64x64** pixels in RGB format (3 channels).


```python
index = 124
plt.imshow(X_train_orig[index]) #display sample training image
plt.show()
```


![png](output_7_0.png)


<a name='2'></a>
## 2 - Layers in TF Keras 

In the previous assignment, you created layers manually in numpy. In TF Keras, you don't have to write code directly to create layers. Rather, TF Keras has pre-defined layers you can use. 

When you create a layer in TF Keras, you are creating a function that takes some input and transforms it into an output you can reuse later. Nice and easy! 

<a name='3'></a>
## 3 - The Sequential API

In the previous assignment, you built helper functions using `numpy` to understand the mechanics behind convolutional neural networks. Most practical applications of deep learning today are built using programming frameworks, which have many built-in functions you can simply call. Keras is a high-level abstraction built on top of TensorFlow, which allows for even more simplified and optimized model creation and training. 

For the first part of this assignment, you'll create a model using TF Keras' Sequential API, which allows you to build layer by layer, and is ideal for building models where each layer has **exactly one** input tensor and **one** output tensor. 

As you'll see, using the Sequential API is simple and straightforward, but is only appropriate for simpler, more straightforward tasks. Later in this notebook you'll spend some time building with a more flexible, powerful alternative: the Functional API. 
 

<a name='3-1'></a>
### 3.1 - Create the Sequential Model

As mentioned earlier, the TensorFlow Keras Sequential API can be used to build simple models with layer operations that proceed in a sequential order. 

You can also add layers incrementally to a Sequential model with the `.add()` method, or remove them using the `.pop()` method, much like you would in a regular Python list.

Actually, you can think of a Sequential model as behaving like a list of layers. Like Python lists, Sequential layers are ordered, and the order in which they are specified matters.  If your model is non-linear or contains layers with multiple inputs or outputs, a Sequential model wouldn't be the right choice!

For any layer construction in Keras, you'll need to specify the input shape in advance. This is because in Keras, the shape of the weights is based on the shape of the inputs. The weights are only created when the model first sees some input data. Sequential models can be created by passing a list of layers to the Sequential constructor, like you will do in the next assignment.

<a name='ex-1'></a>
### Exercise 1 - happyModel

Implement the `happyModel` function below to build the following model: `ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Take help from [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers) 

Also, plug in the following parameters for all the steps:

 - [ZeroPadding2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D): padding 3, input shape 64 x 64 x 3
 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 32 7x7 filters, stride 1
 - [BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization): for axis 3
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Using default parameters
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 1 neuron and a sigmoid activation. 
 
 
 **Hint:**
 
 Use **tfl** as shorthand for **tensorflow.keras.layers**


```python
# GRADED FUNCTION: happyModel

def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    model = tf.keras.Sequential([
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3

            ## Conv2D with 32 7x7 filters and stride of 1
            
            ## BatchNormalization for axis 3
            
            ## ReLU
            
            ## Max Pooling 2D with default parameters
            
            ## Flatten layer
            
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            
            # YOUR CODE STARTS HERE
            tfl.ZeroPadding2D(padding=(3,3), input_shape=(64, 64, 3), data_format="channels_last", name="zeropadding"),
        
            tfl.Conv2D(filters=32, kernel_size = (7,7), strides=(1, 1), \
                        padding='valid',
                        data_format=None,
                        dilation_rate=(1, 1),
                        groups=1,
                        activation=None,
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None,
                        name  = "conv2d"),
        
            tfl.BatchNormalization(axis=3, momentum=0.99, epsilon=0.001, center=True,
                        scale=True,
                        beta_initializer='zeros',
                        gamma_initializer='ones',
                        moving_mean_initializer='zeros',
                        moving_variance_initializer='ones',
                        beta_regularizer=None,
                        gamma_regularizer=None,
                        beta_constraint=None,
                        gamma_constraint=None,
                        name  = "batchnorm"),
        
            tfl.ReLU(max_value=None, negative_slope=0.0, threshold=0.0, name = "relu"),
        
            tfl.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid',
                        data_format=None,
                        name  = "maxpooling"),
        
            tfl.Flatten(data_format=None, name = "flatten"),
        
            tfl.Dense(units = 1, activation = "sigmoid", use_bias=True,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None,
                        name  = "dense")            
            # YOUR CODE ENDS HERE
        ])
    
    return model
```


```python
happy_model = happyModel()
# Print a summary for each layer
for layer in summary(happy_model):
    print(layer)
    
output = [['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
            ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform'],
            ['BatchNormalization', (None, 64, 64, 32), 128],
            ['ReLU', (None, 64, 64, 32), 0],
            ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid'],
            ['Flatten', (None, 32768), 0],
            ['Dense', (None, 1), 32769, 'sigmoid']]
    
comparator(summary(happy_model), output)
```

    ['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))]
    ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform']
    ['BatchNormalization', (None, 64, 64, 32), 128]
    ['ReLU', (None, 64, 64, 32), 0]
    ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid']
    ['Flatten', (None, 32768), 0]
    ['Dense', (None, 1), 32769, 'sigmoid']
    [32mAll tests passed![0m


Now that your model is created, you can compile it for training with an optimizer and loss of your choice. When the string `accuracy` is specified as a metric, the type of accuracy used will be automatically converted based on the loss function used. This is one of the many optimizations built into TensorFlow that make your life easier! If you'd like to read more on how the compiler operates, check the docs [here](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile).


```python
happy_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
```

It's time to check your model's parameters with the `.summary()` method. This will display the types of layers you have, the shape of the outputs, and how many parameters are in each layer. 


```python
happy_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    zeropadding (ZeroPadding2D)  (None, 70, 70, 3)         0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 64, 64, 32)        4736      
    _________________________________________________________________
    batchnorm (BatchNormalizatio (None, 64, 64, 32)        128       
    _________________________________________________________________
    relu (ReLU)                  (None, 64, 64, 32)        0         
    _________________________________________________________________
    maxpooling (MaxPooling2D)    (None, 32, 32, 32)        0         
    _________________________________________________________________
    flatten (Flatten)            (None, 32768)             0         
    _________________________________________________________________
    dense (Dense)                (None, 1)                 32769     
    =================================================================
    Total params: 37,633
    Trainable params: 37,569
    Non-trainable params: 64
    _________________________________________________________________


<a name='3-2'></a>
### 3.2 - Train and Evaluate the Model

After creating the model, compiling it with your choice of optimizer and loss function, and doing a sanity check on its contents, you are now ready to build! 

Simply call `.fit()` to train. That's it! No need for mini-batching, saving, or complex backpropagation computations. That's all been done for you, as you're using a TensorFlow dataset with the batches specified already. You do have the option to specify epoch number or minibatch size if you like (for example, in the case of an un-batched dataset).


```python
happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)
```

    Epoch 1/10
    38/38 [==============================] - 4s 100ms/step - loss: 1.3749 - accuracy: 0.6983
    Epoch 2/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.2642 - accuracy: 0.8900
    Epoch 3/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1668 - accuracy: 0.9183
    Epoch 4/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.2245 - accuracy: 0.9000
    Epoch 5/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1557 - accuracy: 0.9400
    Epoch 6/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1621 - accuracy: 0.9367
    Epoch 7/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1105 - accuracy: 0.9617
    Epoch 8/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.2195 - accuracy: 0.9217
    Epoch 9/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.0966 - accuracy: 0.9617
    Epoch 10/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.1745 - accuracy: 0.9300





    <tensorflow.python.keras.callbacks.History at 0x7faef280e050>



After that completes, just use `.evaluate()` to evaluate against your test set. This function will print the value of the loss function and the performance metrics specified during the compilation of the model. In this case, the `binary_crossentropy` and the `accuracy` respectively.


```python
happy_model.evaluate(X_test, Y_test)
```

    5/5 [==============================] - 0s 34ms/step - loss: 0.1545 - accuracy: 0.9400





    [0.15448236465454102, 0.9399999976158142]



Easy, right? But what if you need to build a model with shared layers, branches, or multiple inputs and outputs? This is where Sequential, with its beautifully simple yet limited functionality, won't be able to help you. 

Next up: Enter the Functional API, your slightly more complex, highly flexible friend.  

<a name='4'></a>
## 4 - The Functional API

Welcome to the second half of the assignment, where you'll use Keras' flexible [Functional API](https://www.tensorflow.org/guide/keras/functional) to build a ConvNet that can differentiate between 6 sign language digits. 

The Functional API can handle models with non-linear topology, shared layers, as well as layers with multiple inputs or outputs. Imagine that, where the Sequential API requires the model to move in a linear fashion through its layers, the Functional API allows much more flexibility. Where Sequential is a straight line, a Functional model is a graph, where the nodes of the layers can connect in many more ways than one. 

In the visual example below, the one possible direction of the movement Sequential model is shown in contrast to a skip connection, which is just one of the many ways a Functional model can be constructed. A skip connection, as you might have guessed, skips some layer in the network and feeds the output to a later layer in the network. Don't worry, you'll be spending more time with skip connections very soon! 

<img src="images/seq_vs_func.png" style="width:350px;height:200px;">

<a name='4-1'></a>
### 4.1 - Load the SIGNS Dataset

As a reminder, the SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5.


```python
# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()
```

<img src="images/SIGNS.png" style="width:800px;height:300px;">

The next cell will show you an example of a labelled image in the dataset. Feel free to change the value of `index` below and re-run to see different examples. 


```python
# Example of an image from the dataset
index = 60
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
```

    y = 4



![png](output_28_1.png)


<a name='4-2'></a>
### 4.2 - Split the Data into Train/Test Sets

In Course 2, you built a fully-connected network for this dataset. But since this is an image dataset, it is more natural to apply a ConvNet to it.

To get started, let's examine the shapes of your data. 


```python
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 1080
    number of test examples = 120
    X_train shape: (1080, 64, 64, 3)
    Y_train shape: (1080, 6)
    X_test shape: (120, 64, 64, 3)
    Y_test shape: (120, 6)


<a name='4-3'></a>
### 4.3 - Forward Propagation

In TensorFlow, there are built-in functions that implement the convolution steps for you. By now, you should be familiar with how TensorFlow builds computational graphs. In the [Functional API](https://www.tensorflow.org/guide/keras/functional), you create a graph of layers. This is what allows such great flexibility.

However, the following model could also be defined using the Sequential API since the information flow is on a single line. But don't deviate. What we want you to learn is to use the functional API.

Begin building your graph of layers by creating an input node that functions as a callable object:

- **input_img = tf.keras.Input(shape=input_shape):** 

Then, create a new node in the graph of layers by calling a layer on the `input_img` object: 

- **tf.keras.layers.Conv2D(filters= ... , kernel_size= ... , padding='same')(input_img):** Read the full documentation on [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D).

- **tf.keras.layers.MaxPool2D(pool_size=(f, f), strides=(s, s), padding='same'):** `MaxPool2D()` downsamples your input using a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window.  For max pooling, you usually operate on a single example at a time and a single channel at a time. Read the full documentation on [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D).

- **tf.keras.layers.ReLU():** computes the elementwise ReLU of Z (which can be any shape). You can read the full documentation on [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU).

- **tf.keras.layers.Flatten()**: given a tensor "P", this function takes each training (or test) example in the batch and flattens it into a 1D vector.  

    * If a tensor P has the shape (batch_size,h,w,c), it returns a flattened tensor with shape (batch_size, k), where $k=h \times w \times c$.  "k" equals the product of all the dimension sizes other than the first dimension.
    
    * For example, given a tensor with dimensions [100, 2, 3, 4], it flattens the tensor to be of shape [100, 24], where 24 = 2 * 3 * 4.  You can read the full documentation on [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten).

- **tf.keras.layers.Dense(units= ... , activation='softmax')(F):** given the flattened input F, it returns the output computed using a fully connected layer. You can read the full documentation on [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense).

In the last function above (`tf.keras.layers.Dense()`), the fully connected layer automatically initializes weights in the graph and keeps on training them as you train the model. Hence, you did not need to initialize those weights when initializing the parameters.

Lastly, before creating the model, you'll need to define the output using the last of the function's compositions (in this example, a Dense layer): 

- **outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)**


#### Window, kernel, filter, pool

The words "kernel" and "filter" are used to refer to the same thing. The word "filter" accounts for the amount of "kernels" that will be used in a single convolution layer. "Pool" is the name of the operation that takes the max or average value of the kernels. 

This is why the parameter `pool_size` refers to `kernel_size`, and you use `(f,f)` to refer to the filter size. 

Pool size and kernel size refer to the same thing in different objects - They refer to the shape of the window where the operation takes place. 

<a name='ex-2'></a>
### Exercise 2 - convolutional_model

Implement the `convolutional_model` function below to build the following model: `CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Use the functions above! 

Also, plug in the following parameters for all the steps:

 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 8 4 by 4 filters, stride 1, padding is "SAME"
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Use an 8 by 8 filter size and an 8 by 8 stride, padding is "SAME"
 - **Conv2D**: Use 16 2 by 2 filters, stride 1, padding is "SAME"
 - **ReLU**
 - **MaxPool2D**: Use a 4 by 4 filter size and a 4 by 4 stride, padding is "SAME"
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 6 neurons and a softmax activation. 


```python
# GRADED FUNCTION: convolutional_model

def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    # Z1 = None
    ## RELU
    # A1 = None
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    # P1 = None
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    # Z2 = None
    ## RELU
    # A2 = None
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    # P2 = None
    ## FLATTEN
    # F = None
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    # outputs = None
    
    # YOUR CODE STARTS HERE
    
    # CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tf.keras.layers.Conv2D(
            filters = 8,
            kernel_size = (4, 4),
            strides=(1, 1),
            padding='same',
            data_format=None,
            dilation_rate=(1, 1),
            groups=1,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name = 'Z1')(input_img)
    
    # RELU
    A1 = tf.keras.layers.ReLU(
            max_value=None, 
            negative_slope=0.0,
            threshold=0.0, 
            name = 'A1')(Z1)
    
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.keras.layers.MaxPool2D(
            pool_size=(8, 8),
            strides= 8,
            padding='same',
            data_format=None,
            name = 'P1')(A1)
    
    # CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tf.keras.layers.Conv2D(
            filters = 16,
            kernel_size = (2, 2),
            strides=(1, 1),
            padding='same',
            data_format=None,
            dilation_rate=(1, 1),
            groups=1,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name = 'Z2')(P1)
    
    # RELU
    A2 = tf.keras.layers.ReLU(
            max_value=None, 
            negative_slope=0.0,
            threshold=0.0, 
            name = 'A2')(Z2)
    
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.keras.layers.MaxPool2D(
            pool_size=(4, 4),
            strides= 4,
            padding='same',
            data_format=None,
            name = 'P2')(A2)
    
    # FLATTEN
    F = tf.keras.layers.Flatten(
            data_format=None,
            name = 'F')(P2)
    
    # Dense layer
    # 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    outputs = tf.keras.layers.Dense(
            units = 6,
            activation= 'softmax',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name = 'outputs')(F)
    
    # YOUR CODE ENDS HERE
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model
```


```python
conv_model = convolutional_model((64, 64, 3))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
conv_model.summary()
    
output = [['InputLayer', [(None, 64, 64, 3)], 0],
        ['Conv2D', (None, 64, 64, 8), 392, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 64, 64, 8), 0],
        ['MaxPooling2D', (None, 8, 8, 8), 0, (8, 8), (8, 8), 'same'],
        ['Conv2D', (None, 8, 8, 16), 528, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 8, 8, 16), 0],
        ['MaxPooling2D', (None, 2, 2, 16), 0, (4, 4), (4, 4), 'same'],
        ['Flatten', (None, 64), 0],
        ['Dense', (None, 6), 390, 'softmax']]
    
comparator(summary(conv_model), output)
```

    Model: "functional_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 64, 64, 3)]       0         
    _________________________________________________________________
    Z1 (Conv2D)                  (None, 64, 64, 8)         392       
    _________________________________________________________________
    A1 (ReLU)                    (None, 64, 64, 8)         0         
    _________________________________________________________________
    P1 (MaxPooling2D)            (None, 8, 8, 8)           0         
    _________________________________________________________________
    Z2 (Conv2D)                  (None, 8, 8, 16)          528       
    _________________________________________________________________
    A2 (ReLU)                    (None, 8, 8, 16)          0         
    _________________________________________________________________
    P2 (MaxPooling2D)            (None, 2, 2, 16)          0         
    _________________________________________________________________
    F (Flatten)                  (None, 64)                0         
    _________________________________________________________________
    outputs (Dense)              (None, 6)                 390       
    =================================================================
    Total params: 1,310
    Trainable params: 1,310
    Non-trainable params: 0
    _________________________________________________________________
    [32mAll tests passed![0m


Both the Sequential and Functional APIs return a TF Keras model object. The only difference is how inputs are handled inside the object model! 

<a name='4-4'></a>
### 4.4 - Train the Model


```python
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)
```

    Epoch 1/100
    17/17 [==============================] - 2s 112ms/step - loss: 1.7977 - accuracy: 0.1981 - val_loss: 1.7884 - val_accuracy: 0.2417
    Epoch 2/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.7860 - accuracy: 0.2435 - val_loss: 1.7833 - val_accuracy: 0.1917
    Epoch 3/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7805 - accuracy: 0.2694 - val_loss: 1.7791 - val_accuracy: 0.2917
    Epoch 4/100
    17/17 [==============================] - 2s 112ms/step - loss: 1.7752 - accuracy: 0.3444 - val_loss: 1.7754 - val_accuracy: 0.3500
    Epoch 5/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.7690 - accuracy: 0.3713 - val_loss: 1.7709 - val_accuracy: 0.3583
    Epoch 6/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7589 - accuracy: 0.3963 - val_loss: 1.7639 - val_accuracy: 0.3417
    Epoch 7/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.7431 - accuracy: 0.4111 - val_loss: 1.7528 - val_accuracy: 0.3500
    Epoch 8/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7207 - accuracy: 0.4204 - val_loss: 1.7367 - val_accuracy: 0.3083
    Epoch 9/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.6914 - accuracy: 0.4333 - val_loss: 1.7159 - val_accuracy: 0.3333
    Epoch 10/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.6555 - accuracy: 0.4398 - val_loss: 1.6894 - val_accuracy: 0.3750
    Epoch 11/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.6131 - accuracy: 0.4444 - val_loss: 1.6575 - val_accuracy: 0.3667
    Epoch 12/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.5667 - accuracy: 0.4620 - val_loss: 1.6213 - val_accuracy: 0.3417
    Epoch 13/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.5178 - accuracy: 0.4694 - val_loss: 1.5809 - val_accuracy: 0.4000
    Epoch 14/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.4656 - accuracy: 0.4704 - val_loss: 1.5311 - val_accuracy: 0.4167
    Epoch 15/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.4074 - accuracy: 0.4963 - val_loss: 1.4766 - val_accuracy: 0.4417
    Epoch 16/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.3541 - accuracy: 0.5204 - val_loss: 1.4276 - val_accuracy: 0.4583
    Epoch 17/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.3053 - accuracy: 0.5519 - val_loss: 1.3819 - val_accuracy: 0.5000
    Epoch 18/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.2579 - accuracy: 0.5657 - val_loss: 1.3353 - val_accuracy: 0.5083
    Epoch 19/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.2134 - accuracy: 0.5880 - val_loss: 1.2913 - val_accuracy: 0.5167
    Epoch 20/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.1707 - accuracy: 0.6037 - val_loss: 1.2486 - val_accuracy: 0.5583
    Epoch 21/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.1305 - accuracy: 0.6259 - val_loss: 1.2106 - val_accuracy: 0.5833
    Epoch 22/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.0923 - accuracy: 0.6343 - val_loss: 1.1711 - val_accuracy: 0.6000
    Epoch 23/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.0565 - accuracy: 0.6537 - val_loss: 1.1386 - val_accuracy: 0.6250
    Epoch 24/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.0226 - accuracy: 0.6639 - val_loss: 1.1049 - val_accuracy: 0.6417
    Epoch 25/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.9908 - accuracy: 0.6759 - val_loss: 1.0748 - val_accuracy: 0.6417
    Epoch 26/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.9606 - accuracy: 0.6778 - val_loss: 1.0480 - val_accuracy: 0.6417
    Epoch 27/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.9331 - accuracy: 0.6935 - val_loss: 1.0225 - val_accuracy: 0.6500
    Epoch 28/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.9062 - accuracy: 0.6991 - val_loss: 0.9962 - val_accuracy: 0.6583
    Epoch 29/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8821 - accuracy: 0.7093 - val_loss: 0.9745 - val_accuracy: 0.6583
    Epoch 30/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8589 - accuracy: 0.7120 - val_loss: 0.9519 - val_accuracy: 0.6667
    Epoch 31/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8375 - accuracy: 0.7204 - val_loss: 0.9315 - val_accuracy: 0.6667
    Epoch 32/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8172 - accuracy: 0.7287 - val_loss: 0.9120 - val_accuracy: 0.6833
    Epoch 33/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7978 - accuracy: 0.7352 - val_loss: 0.8931 - val_accuracy: 0.6667
    Epoch 34/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.7800 - accuracy: 0.7398 - val_loss: 0.8757 - val_accuracy: 0.6833
    Epoch 35/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7635 - accuracy: 0.7398 - val_loss: 0.8607 - val_accuracy: 0.7000
    Epoch 36/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7480 - accuracy: 0.7574 - val_loss: 0.8453 - val_accuracy: 0.7083
    Epoch 37/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7325 - accuracy: 0.7611 - val_loss: 0.8321 - val_accuracy: 0.7083
    Epoch 38/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7186 - accuracy: 0.7657 - val_loss: 0.8179 - val_accuracy: 0.7083
    Epoch 39/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7051 - accuracy: 0.7685 - val_loss: 0.8063 - val_accuracy: 0.7167
    Epoch 40/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.6923 - accuracy: 0.7731 - val_loss: 0.7938 - val_accuracy: 0.7167
    Epoch 41/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6800 - accuracy: 0.7769 - val_loss: 0.7830 - val_accuracy: 0.7167
    Epoch 42/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.6680 - accuracy: 0.7833 - val_loss: 0.7728 - val_accuracy: 0.7167
    Epoch 43/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6570 - accuracy: 0.7870 - val_loss: 0.7623 - val_accuracy: 0.7167
    Epoch 44/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.6461 - accuracy: 0.7889 - val_loss: 0.7529 - val_accuracy: 0.7167
    Epoch 45/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6362 - accuracy: 0.7926 - val_loss: 0.7441 - val_accuracy: 0.7333
    Epoch 46/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6261 - accuracy: 0.7963 - val_loss: 0.7353 - val_accuracy: 0.7417
    Epoch 47/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6167 - accuracy: 0.7972 - val_loss: 0.7269 - val_accuracy: 0.7417
    Epoch 48/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6073 - accuracy: 0.8009 - val_loss: 0.7189 - val_accuracy: 0.7417
    Epoch 49/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5983 - accuracy: 0.8046 - val_loss: 0.7114 - val_accuracy: 0.7417
    Epoch 50/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5900 - accuracy: 0.8037 - val_loss: 0.7037 - val_accuracy: 0.7417
    Epoch 51/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.5818 - accuracy: 0.8074 - val_loss: 0.6954 - val_accuracy: 0.7500
    Epoch 52/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5742 - accuracy: 0.8130 - val_loss: 0.6880 - val_accuracy: 0.7583
    Epoch 53/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5664 - accuracy: 0.8148 - val_loss: 0.6806 - val_accuracy: 0.7500
    Epoch 54/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5593 - accuracy: 0.8157 - val_loss: 0.6739 - val_accuracy: 0.7583
    Epoch 55/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5518 - accuracy: 0.8176 - val_loss: 0.6678 - val_accuracy: 0.7500
    Epoch 56/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5451 - accuracy: 0.8185 - val_loss: 0.6611 - val_accuracy: 0.7583
    Epoch 57/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.5385 - accuracy: 0.8204 - val_loss: 0.6549 - val_accuracy: 0.7500
    Epoch 58/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.5325 - accuracy: 0.8241 - val_loss: 0.6485 - val_accuracy: 0.7500
    Epoch 59/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.5255 - accuracy: 0.8287 - val_loss: 0.6420 - val_accuracy: 0.7667
    Epoch 60/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5191 - accuracy: 0.8306 - val_loss: 0.6371 - val_accuracy: 0.7667
    Epoch 61/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5126 - accuracy: 0.8315 - val_loss: 0.6322 - val_accuracy: 0.7750
    Epoch 62/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5070 - accuracy: 0.8315 - val_loss: 0.6264 - val_accuracy: 0.7750
    Epoch 63/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5012 - accuracy: 0.8343 - val_loss: 0.6212 - val_accuracy: 0.7750
    Epoch 64/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.4957 - accuracy: 0.8380 - val_loss: 0.6164 - val_accuracy: 0.7750
    Epoch 65/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4900 - accuracy: 0.8407 - val_loss: 0.6109 - val_accuracy: 0.7750
    Epoch 66/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.4846 - accuracy: 0.8435 - val_loss: 0.6058 - val_accuracy: 0.7750
    Epoch 67/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4790 - accuracy: 0.8444 - val_loss: 0.6017 - val_accuracy: 0.7750
    Epoch 68/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4736 - accuracy: 0.8463 - val_loss: 0.5962 - val_accuracy: 0.7833
    Epoch 69/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.4680 - accuracy: 0.8491 - val_loss: 0.5913 - val_accuracy: 0.7833
    Epoch 70/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4626 - accuracy: 0.8500 - val_loss: 0.5854 - val_accuracy: 0.7917
    Epoch 71/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4575 - accuracy: 0.8500 - val_loss: 0.5813 - val_accuracy: 0.7917
    Epoch 72/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4526 - accuracy: 0.8509 - val_loss: 0.5767 - val_accuracy: 0.7917
    Epoch 73/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4479 - accuracy: 0.8528 - val_loss: 0.5723 - val_accuracy: 0.7917
    Epoch 74/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.4431 - accuracy: 0.8537 - val_loss: 0.5673 - val_accuracy: 0.7917
    Epoch 75/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.4386 - accuracy: 0.8556 - val_loss: 0.5634 - val_accuracy: 0.7917
    Epoch 76/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4343 - accuracy: 0.8556 - val_loss: 0.5587 - val_accuracy: 0.7917
    Epoch 77/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4299 - accuracy: 0.8574 - val_loss: 0.5549 - val_accuracy: 0.7917
    Epoch 78/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4256 - accuracy: 0.8574 - val_loss: 0.5509 - val_accuracy: 0.7917
    Epoch 79/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.4214 - accuracy: 0.8602 - val_loss: 0.5471 - val_accuracy: 0.7917
    Epoch 80/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.4171 - accuracy: 0.8611 - val_loss: 0.5441 - val_accuracy: 0.7917
    Epoch 81/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4128 - accuracy: 0.8639 - val_loss: 0.5407 - val_accuracy: 0.8000
    Epoch 82/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4087 - accuracy: 0.8657 - val_loss: 0.5370 - val_accuracy: 0.8083
    Epoch 83/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4046 - accuracy: 0.8639 - val_loss: 0.5340 - val_accuracy: 0.8167
    Epoch 84/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4006 - accuracy: 0.8639 - val_loss: 0.5314 - val_accuracy: 0.8167
    Epoch 85/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3964 - accuracy: 0.8639 - val_loss: 0.5281 - val_accuracy: 0.8250
    Epoch 86/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.3926 - accuracy: 0.8685 - val_loss: 0.5244 - val_accuracy: 0.8167
    Epoch 87/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3886 - accuracy: 0.8722 - val_loss: 0.5209 - val_accuracy: 0.8167
    Epoch 88/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3854 - accuracy: 0.8713 - val_loss: 0.5181 - val_accuracy: 0.8167
    Epoch 89/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.3818 - accuracy: 0.8741 - val_loss: 0.5151 - val_accuracy: 0.8167
    Epoch 90/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3781 - accuracy: 0.8759 - val_loss: 0.5120 - val_accuracy: 0.8250
    Epoch 91/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3745 - accuracy: 0.8778 - val_loss: 0.5089 - val_accuracy: 0.8250
    Epoch 92/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3709 - accuracy: 0.8787 - val_loss: 0.5061 - val_accuracy: 0.8167
    Epoch 93/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.3675 - accuracy: 0.8806 - val_loss: 0.5036 - val_accuracy: 0.8167
    Epoch 94/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3640 - accuracy: 0.8806 - val_loss: 0.5000 - val_accuracy: 0.8167
    Epoch 95/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3608 - accuracy: 0.8815 - val_loss: 0.4964 - val_accuracy: 0.8167
    Epoch 96/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.3576 - accuracy: 0.8815 - val_loss: 0.4939 - val_accuracy: 0.8167
    Epoch 97/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.3546 - accuracy: 0.8843 - val_loss: 0.4914 - val_accuracy: 0.8333
    Epoch 98/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3515 - accuracy: 0.8852 - val_loss: 0.4880 - val_accuracy: 0.8333
    Epoch 99/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3482 - accuracy: 0.8852 - val_loss: 0.4861 - val_accuracy: 0.8333
    Epoch 100/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.3450 - accuracy: 0.8843 - val_loss: 0.4826 - val_accuracy: 0.8333


<a name='5'></a>
## 5 - History Object 

The history object is an output of the `.fit()` operation, and provides a record of all the loss and metric values in memory. It's stored as a dictionary that you can retrieve at `history.history`: 


```python
history.history
```




    {'loss': [1.7977429628372192,
      1.78599214553833,
      1.780454158782959,
      1.775235652923584,
      1.7689677476882935,
      1.7589219808578491,
      1.7430928945541382,
      1.7206947803497314,
      1.6913983821868896,
      1.6555240154266357,
      1.6131422519683838,
      1.56667160987854,
      1.517799735069275,
      1.465559959411621,
      1.407373309135437,
      1.3541488647460938,
      1.305289626121521,
      1.257947325706482,
      1.2133647203445435,
      1.170694351196289,
      1.1305094957351685,
      1.0922728776931763,
      1.0565006732940674,
      1.0226049423217773,
      0.9908023476600647,
      0.9605674743652344,
      0.9330645799636841,
      0.9062013626098633,
      0.8820641040802002,
      0.858946681022644,
      0.8375148773193359,
      0.8171613216400146,
      0.7978479862213135,
      0.7800052165985107,
      0.763480007648468,
      0.7479587197303772,
      0.7325338125228882,
      0.718597412109375,
      0.7051306962966919,
      0.6923219561576843,
      0.6799935698509216,
      0.6680055856704712,
      0.6570293307304382,
      0.6460748910903931,
      0.636175811290741,
      0.6260632276535034,
      0.6166794300079346,
      0.6073323488235474,
      0.5983165502548218,
      0.5899888873100281,
      0.5818026661872864,
      0.5741680860519409,
      0.5664381384849548,
      0.5592623353004456,
      0.551813006401062,
      0.5451142191886902,
      0.538493812084198,
      0.5325103998184204,
      0.5255469679832458,
      0.5191047787666321,
      0.5126068592071533,
      0.5070277452468872,
      0.5012215375900269,
      0.495738685131073,
      0.4899974763393402,
      0.4846048951148987,
      0.47900447249412537,
      0.47362038493156433,
      0.4680129885673523,
      0.4625878930091858,
      0.4574715495109558,
      0.4526485204696655,
      0.44786766171455383,
      0.44313758611679077,
      0.4386066496372223,
      0.4342885911464691,
      0.42994919419288635,
      0.4255867600440979,
      0.42143696546554565,
      0.417057603597641,
      0.4128122925758362,
      0.40868711471557617,
      0.4045947194099426,
      0.4005575478076935,
      0.3964371383190155,
      0.3925827443599701,
      0.3886474370956421,
      0.3853658139705658,
      0.381803959608078,
      0.3781181275844574,
      0.3744828999042511,
      0.37088775634765625,
      0.3674604892730713,
      0.3639835715293884,
      0.3607643246650696,
      0.3576275110244751,
      0.35461193323135376,
      0.3515486419200897,
      0.34823256731033325,
      0.34498152136802673],
     'accuracy': [0.19814814627170563,
      0.24351851642131805,
      0.26944443583488464,
      0.3444444537162781,
      0.3712962865829468,
      0.39629629254341125,
      0.41111111640930176,
      0.4203703701496124,
      0.4333333373069763,
      0.43981480598449707,
      0.4444444477558136,
      0.46203702688217163,
      0.4694444537162781,
      0.4703703820705414,
      0.4962962865829468,
      0.520370364189148,
      0.5518518686294556,
      0.5657407641410828,
      0.5879629850387573,
      0.6037036776542664,
      0.6259258985519409,
      0.6342592835426331,
      0.6537036895751953,
      0.6638888716697693,
      0.6759259104728699,
      0.6777777671813965,
      0.6935185194015503,
      0.6990740895271301,
      0.7092592716217041,
      0.7120370268821716,
      0.720370352268219,
      0.7287036776542664,
      0.7351852059364319,
      0.739814817905426,
      0.739814817905426,
      0.7574074268341064,
      0.7611111402511597,
      0.7657407522201538,
      0.7685185074806213,
      0.7731481194496155,
      0.7768518328666687,
      0.7833333611488342,
      0.7870370149612427,
      0.7888888716697693,
      0.7925925850868225,
      0.7962962985038757,
      0.7972221970558167,
      0.8009259104728699,
      0.8046296238899231,
      0.8037037253379822,
      0.8074073791503906,
      0.8129629492759705,
      0.8148148059844971,
      0.8157407641410828,
      0.8175926208496094,
      0.8185185194015503,
      0.8203703761100769,
      0.8240740895271301,
      0.8287037014961243,
      0.8305555582046509,
      0.8314814567565918,
      0.8314814567565918,
      0.8342592716217041,
      0.8379629850387573,
      0.8407407402992249,
      0.8435184955596924,
      0.8444444537162781,
      0.8462963104248047,
      0.8490740656852722,
      0.8500000238418579,
      0.8500000238418579,
      0.8509259223937988,
      0.8527777791023254,
      0.8537036776542664,
      0.855555534362793,
      0.855555534362793,
      0.8574073910713196,
      0.8574073910713196,
      0.8601852059364319,
      0.8611111044883728,
      0.8638888597488403,
      0.8657407164573669,
      0.8638888597488403,
      0.8638888597488403,
      0.8638888597488403,
      0.8685185313224792,
      0.8722222447395325,
      0.8712962865829468,
      0.8740741014480591,
      0.8759258985519409,
      0.8777777552604675,
      0.8787037134170532,
      0.8805555701255798,
      0.8805555701255798,
      0.8814814686775208,
      0.8814814686775208,
      0.8842592835426331,
      0.885185182094574,
      0.885185182094574,
      0.8842592835426331],
     'val_loss': [1.7883814573287964,
      1.7833491563796997,
      1.7790967226028442,
      1.7753616571426392,
      1.7708724737167358,
      1.7639427185058594,
      1.7527801990509033,
      1.7367088794708252,
      1.7158905267715454,
      1.6894147396087646,
      1.657452940940857,
      1.6213313341140747,
      1.580871820449829,
      1.5311470031738281,
      1.4766100645065308,
      1.4275919198989868,
      1.381885290145874,
      1.3353469371795654,
      1.2912633419036865,
      1.2485737800598145,
      1.2106341123580933,
      1.1711126565933228,
      1.1386337280273438,
      1.1048693656921387,
      1.0748010873794556,
      1.0479793548583984,
      1.022495150566101,
      0.9962356686592102,
      0.974457323551178,
      0.9519386887550354,
      0.9314906001091003,
      0.9119547009468079,
      0.8930807709693909,
      0.8757328391075134,
      0.8606825470924377,
      0.8452866673469543,
      0.832111656665802,
      0.8178787231445312,
      0.8063033819198608,
      0.7938443422317505,
      0.7829583287239075,
      0.7727594971656799,
      0.7622940540313721,
      0.7528874278068542,
      0.7441035509109497,
      0.7352616190910339,
      0.7269162535667419,
      0.7189103364944458,
      0.7113717198371887,
      0.7036965489387512,
      0.6953957676887512,
      0.6879777908325195,
      0.680590808391571,
      0.6738523840904236,
      0.6677539348602295,
      0.6611114740371704,
      0.6549415588378906,
      0.6485411524772644,
      0.641950249671936,
      0.6371117234230042,
      0.632191002368927,
      0.626438558101654,
      0.6211627721786499,
      0.6163906455039978,
      0.610927939414978,
      0.6058429479598999,
      0.6017242670059204,
      0.5962290167808533,
      0.5913353562355042,
      0.5854021906852722,
      0.5812535881996155,
      0.5767216086387634,
      0.5722599625587463,
      0.5673466324806213,
      0.5633620023727417,
      0.5586525201797485,
      0.554922878742218,
      0.5509417057037354,
      0.5471164584159851,
      0.5440889596939087,
      0.5406786203384399,
      0.5369855761528015,
      0.533969521522522,
      0.5313713550567627,
      0.5281423926353455,
      0.5244024395942688,
      0.5208941102027893,
      0.5180864930152893,
      0.51512211561203,
      0.5120199918746948,
      0.5088707804679871,
      0.5060670971870422,
      0.5035702586174011,
      0.5000258088111877,
      0.49637696146965027,
      0.4939139187335968,
      0.49142733216285706,
      0.48797595500946045,
      0.4860748052597046,
      0.4825999140739441],
     'val_accuracy': [0.24166665971279144,
      0.19166666269302368,
      0.2916666567325592,
      0.3499999940395355,
      0.3583333194255829,
      0.34166666865348816,
      0.3499999940395355,
      0.3083333373069763,
      0.3333333432674408,
      0.375,
      0.36666667461395264,
      0.34166666865348816,
      0.4000000059604645,
      0.4166666567325592,
      0.4416666626930237,
      0.4583333432674408,
      0.5,
      0.5083333253860474,
      0.5166666507720947,
      0.5583333373069763,
      0.5833333134651184,
      0.6000000238418579,
      0.625,
      0.6416666507720947,
      0.6416666507720947,
      0.6416666507720947,
      0.6499999761581421,
      0.6583333611488342,
      0.6583333611488342,
      0.6666666865348816,
      0.6666666865348816,
      0.6833333373069763,
      0.6666666865348816,
      0.6833333373069763,
      0.699999988079071,
      0.7083333134651184,
      0.7083333134651184,
      0.7083333134651184,
      0.7166666388511658,
      0.7166666388511658,
      0.7166666388511658,
      0.7166666388511658,
      0.7166666388511658,
      0.7166666388511658,
      0.7333333492279053,
      0.7416666746139526,
      0.7416666746139526,
      0.7416666746139526,
      0.7416666746139526,
      0.7416666746139526,
      0.75,
      0.7583333253860474,
      0.75,
      0.7583333253860474,
      0.75,
      0.7583333253860474,
      0.75,
      0.75,
      0.7666666507720947,
      0.7666666507720947,
      0.7749999761581421,
      0.7749999761581421,
      0.7749999761581421,
      0.7749999761581421,
      0.7749999761581421,
      0.7749999761581421,
      0.7749999761581421,
      0.7833333611488342,
      0.7833333611488342,
      0.7916666865348816,
      0.7916666865348816,
      0.7916666865348816,
      0.7916666865348816,
      0.7916666865348816,
      0.7916666865348816,
      0.7916666865348816,
      0.7916666865348816,
      0.7916666865348816,
      0.7916666865348816,
      0.7916666865348816,
      0.800000011920929,
      0.8083333373069763,
      0.8166666626930237,
      0.8166666626930237,
      0.824999988079071,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.824999988079071,
      0.824999988079071,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8333333134651184,
      0.8333333134651184,
      0.8333333134651184,
      0.8333333134651184]}



Now visualize the loss over time using `history.history`: 


```python
# The history.history["loss"] entry is a dictionary with as many values as epochs that the
# model was trained on. 
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
```




    [Text(0, 0.5, 'Accuracy'), Text(0.5, 0, 'Epoch')]




![image](./output_41_1.png)



![image](./output_41_2.png)


**Congratulations**! You've finished the assignment and built two models: One that recognizes  smiles, and another that recognizes SIGN language with almost 80% accuracy on the test set. In addition to that, you now also understand the applications of two Keras APIs: Sequential and Functional. Nicely done! 

By now, you know a bit about how the Functional API works and may have glimpsed the possibilities. In your next assignment, you'll really get a feel for its power when you get the opportunity to build a very deep ConvNet, using ResNets! 

<a name='6'></a>
## 6 - Bibliography

You're always encouraged to read the official documentation. To that end, you can find the docs for the Sequential and Functional APIs here: 

https://www.tensorflow.org/guide/keras/sequential_model

https://www.tensorflow.org/guide/keras/functional
