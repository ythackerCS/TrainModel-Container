import os 
import pandas as pd #for data analysis 
import matplotlib.pyplot as plt 
import cv2 
import numpy as np 
import math 
import pydicom as pydicom
import tensorflow as tf 
import tensorflow_addons as tfa
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


from tensorflow import keras

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model

from tensorflow.keras import preprocessing
from tensorflow.keras import models



#this is a test script to see if model will load given a path to a model. It is not part of the docker pipleine just provided for testing purposes 
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def load(modelName):
#     path = "/var/lib/docker/radioGraphProject/savedModels/" + modelName
    path = "/saveLocation/{0}".format(modelName)
    dependencies = {
        'f1_m': f1_m, 
        'precision_m': precision_m
    }
    print("loading...", path)
    model = tf.keras.models.load_model(filepath = path, custom_objects=dependencies)
    return model



#change the model name below to test loading of your model 
model = load("2021-07-12-val_acc:savedWithoutAcc")
model.summary()
