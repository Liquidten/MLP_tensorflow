#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 00:21:41 2019

@author: sameepshah
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


def Model(train_images,train_labels):
    
    model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28,28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
            ])
    
    return model

def main():   
    
    #import data
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images.shape
    len(train_labels)
    #train_labels
    test_images.shape
    len(test_labels)

    plt.figure()
    plt.imshow(train_images[1])
    plt.colorbar()
    plt.grid(False)
    plt.show

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    '''
    plt.figure(figsize=(10,10))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()
    '''
    
    model = Model(train_images,train_labels)
    model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
    model.fit(train_images, train_labels, epochs = 5)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    
    predictions = model.predict(test_images)
    #print(predictions[2])
    np.argmax(predictions[2])
    #print(test_labels[2])
    
    
main()