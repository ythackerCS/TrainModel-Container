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
from tqdm import tqdm
import argparse
import datetime 


from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
from tensorflow.keras import models


##--Flags for running script in XNAT--## 

parser = argparse.ArgumentParser(description='This script can train and test a covid classifier model from scratch or test/train a model provided')

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError("{0} is not a valid boolean value".format(value))


#required arguments
parser.add_argument('-s',"--save", help="Save the in the 'modelFiles' folder after work is complete?", type=str_to_bool, nargs='?', const=True, default=False, required=True)
parser.add_argument('-t',"--train", help="Should the model be trained", type=str_to_bool, nargs='?', const=True, default=False, required=True)
parser.add_argument('-u',"--underSmp", help="Should the data be undersampled, if false it will be over sampled", type=str_to_bool, nargs='?', const=True, default=False, required=True)

    #optional arguments#
parser.add_argument('-d',"--dataset", help="Name Of dataset to use for training (dont need .csv) if none provided mortality will be used as default", nargs='?', default="mortality", type=str)
parser.add_argument('-n',"--model_name", help="If you provide the name of the model located in the 'modelFiles' folder it will train/test using that model as the base", nargs='?', type=str)
parser.add_argument('-e',"--epochs", help="How many epocs for training (30 is default)", nargs='?', default=30, type=int)
parser.add_argument('-b',"--batch_size", help="What batch size to use for training/testing (10 is default)", nargs='?', default=10, type=int)
parser.add_argument('-p',"--pxl_size", help="Pixel size for image used for training/testing (50 is default)", nargs='?', default=50, type=int)
parser.add_argument('-q',"--test_size", help="Percentage of data that should be used for testing as a percentage decimal i.e 0.3 = 30 percent ('0.3' ~ 30 percent is default)", nargs='?', const=0.3, type=float)
parser.add_argument('-r',"--rand_state", help="Random state of spliting data (50 is default)", nargs='?', default=50, type=int)

    ##monitoring callbacks also optional##
parser.add_argument('-v',"--val_loss_monitor", help="Monitor val_loss for certain patients, if no value provided it will not be monitored", nargs='?', type=int)
parser.add_argument('-l',"--loss_monitor", help="Monitor loss for certain patients, if no value provided it will not be monitored", nargs='?', type=int)

arguments = parser.parse_args()
print(arguments)

## ---------- end argument information ----------- ##


#loading csv from csvlocation folder 
df_train = pd.read_csv('/csvLocation/{0}.csv'.format(arguments.dataset))


##-- UTLITIES --##
    ##-statistic utlities-##
##---Statsitics functions---#
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
#--------------- End statistics function -----------------#


##----------- functional save/load ulitities ----------##

#this function will save the file to a folder called saveLocation within the container 
def saveModel(model, acc):
    now = datetime.datetime.now()
    modelName = str(now.date()).strip()+"-val_acc:"+acc.strip()
    modelName = modelName.strip()
    print("saving...", modelName)
    save_location = "/saveLocation/{0}-{1}".format(modelName,arguments.dataset)
    print(save_location)
    tf.keras.models.save_model(model, filepath=save_location)
    
#this function will load a model from saveLocation within the container 
def load(modelName):
    path = "/saveLocation/{0}".format(modelName)
    #specific metrics if used will need to be added here in my case f1_m and precission_m were used due to imbalanced datasets (prior to balancing being added)
    dependencies = {
        'f1_m': f1_m, 
        'precision_m': precision_m
    }
    print("loading...", path)
    model = tf.keras.models.load_model(filepath = path, custom_objects=dependencies)
    model.compile(metrics=['acc',f1_m,precision_m])
    return model

## -------- End functional utlities ------##



## ---------- Data set up functions, balancing data, and spliting dataset ------------ ######

#this function extracts the pixel_data and resizes it to a user defined size defult is 50, and the associated vector for classification is generated 
def getImagesAndLabels(imageArray, labelArray, img_px_size=50, visualize=False):
    images = []
    labels = []
    uids = []
    idx = 0
    print("getting images and labels")

    for file, mortality in tqdm(zip(imageArray.iteritems(), labelArray.iteritems()),total = len(imageArray)): 
        uid = file[1]
        if mortality[1] == 0: 
            label = [1,0]
        if mortality[1] == 1: 
            label = [0,1]
        path = uid
        image = pydicom.read_file(path)
        if "PixelData" in image: 
            if len(image.pixel_array.shape) == 2: 
                # print(image.pixel_array)
                idx += 1
                resized_image = cv2.resize(np.array(image.pixel_array),(img_px_size,img_px_size))

                #this set of code is commented out as visuilization is not possible within a docker container but if you run this seperatly or within jupyter you are able to visulize every 50th image, change 50 to what ever number you like if you want to visulize more or less 

                # if visualize: 
                #     if idx%50==0:
                #         fig = plt.figure() 
                #         plt.imshow(resized_image)
                #         plt.show()
                        
                images.append(resized_image)
                labels.append(label)
                uids.append(uid)
    print("total subjects avilable: ", idx)
    return images, labels, uids


#as the dataset was imbalanced, balancing tehniques were applied, in this case the number of dicom for each class is counted and then balanced according to user's preference, it can either be undersampeled or over sampeled 
def balanceData(imageArray, labelArray, underSample = False,):
#     print(imageArray, labelArray)
    concatinatedArrray = pd.concat([imageArray, labelArray], axis=1)
    
    count_class_0, count_class_1 = concatinatedArrray.mortality.value_counts()
    df_class_0 = concatinatedArrray[concatinatedArrray['mortality'] == 0]
    df_class_1 = concatinatedArrray[concatinatedArrray['mortality'] == 1]
    
    print("alive", len(df_class_0), "dead", len(df_class_1))
    
#     print("before balancing")
    concatinatedArrray.mortality.value_counts().plot(kind='bar', title='before balancing');
    

    #undersampleling of data is done if user cooses to under sample 
    if arguments.underSmp: 
        df_class_0_under = df_class_0.sample(count_class_1)
        df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

        print('Random under-sampling:')
#         print(df_test_under.mortality.value_counts())

#         print("after balancing")
        df_test_under.mortality.value_counts().plot(kind='bar', title='after balancing_undersample');
        total_data = pd.concat([df_class_0_under, df_class_1])
        
#         print(len(total_data))
        
    #over sampleing is done if user does not check undersample 
    else: 
        df_class_1_over = df_class_1.sample(count_class_0, replace=True)
        df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

        print('Random over-sampling:')
#         print(df_test_over.mortality.value_counts())

#         print("after balancing")
        df_test_over.mortality.value_counts().plot(kind='bar', title='after balancing_oversample');
        total_data = pd.concat([df_class_0, df_class_1_over])
        
#         print(len(total_data))
        
    return total_data.path, total_data.mortality, total_data
        
#this function will split the data in to train,validation, and test datasets steps are as follows: 
    #1 user provides testSize, which will split the orgninal data set in to 1-x% training and x% "test dataset"
    #2 the "test dataset" is then split again in half for validation and half an actual test dataset 

def splitData(px_size, visulize = False, testSize = 0.30, randState = 50, underSamp=False):
    count_class_0, count_class_1 = df_train.mortality.value_counts()
    
    
    df_class_0 = df_train[df_train['mortality'] == 0]
    df_class_1 = df_train[df_train['mortality'] == 1]
    
    
    total_data = pd.concat([df_class_0, df_class_1])
#     print("total", len(total_data))
    images = total_data.path
    labels = total_data.mortality
    
    
    #train and "test dataset" created here 
    image_train, image_test, label_train, label_test = train_test_split(images, labels, test_size=testSize, random_state=randState)
    

    #"test dataset" is split in half for a validation set and a actual test set 
    image_val, image_test, label_val, label_test = train_test_split(image_test, label_test, test_size=0.5, random_state=randState)
    
    #now that data is split we want to balance it
    image_train, label_train, total_data_train = balanceData(image_train, label_train, underSample = underSamp)
   
    image_test, label_test, total_data_test = balanceData(image_test, label_test, underSample = underSamp)
    
    image_val, label_val, total_data_val = balanceData(image_val, label_val, underSample = underSamp)
    

    #statistics of datasets is printed for user convencience 
    print("image train and label train sizes", len(image_train), len(label_train))
    df_class_0 = total_data_train[total_data_train['mortality'] == 0]
    df_class_1 = total_data_train[total_data_train['mortality'] == 1]
    
    print("alive", len(df_class_0), "dead", len(df_class_1))
    
    print("image test and label test sizes", len(image_test), len(label_test))
    df_class_0 = total_data_test[total_data_test['mortality'] == 0]
    df_class_1 = total_data_test[total_data_test['mortality'] == 1]
    
    print("alive", len(df_class_0), "dead", len(df_class_1))

    common_val1 = total_data_train.merge(total_data_test, how = 'inner' ,indicator=False)
    common_values = common_val1.merge(total_data_val, how = 'inner' ,indicator=False).to_numpy()
    print("COMMON", common_values)
    print("COMMON", len(common_values))

    #once datasets are split images and labels are gathered in to a arrays that can be fed into keras training 
    image_train, label_train, uids_train = getImagesAndLabels(image_train, label_train, img_px_size=px_size, visualize=visulize)
    image_test, label_test, uids_test = getImagesAndLabels(image_test, label_test, img_px_size=px_size, visualize=visulize)
    image_val, label_val, uids_val = getImagesAndLabels(image_val, label_val, img_px_size=px_size, visualize=visulize)

    return image_train, image_test, image_val, label_train, label_test, label_val, uids_train, uids_test, uids_val

## ------------------ END data ulities -----------------------##


## ---- MODEL architecture (CNN) ----## 
#this was obtained and developed with the help of a gradstudent, the paper used is: Isensee F, Kickingereder P, Wick W, Bendszus M, Maier-Hein KH. Brain tumor segmentation and radiomics survival prediction: Contribution to the brats 2017 challenge. In: International MICCAI Brainlesion Workshop. Springer; 2017. p. 287–97.
def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, regularizer = None):
    conv1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters, regularizer = regularizer)
    dropout = tf.keras.layers.SpatialDropout2D(rate=dropout_rate)(conv1)
    conv2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters, regularizer = regularizer)
    return conv2

def create_convolution_block(input_layer, n_filters, name=None, kernel=3, padding='SAME', strides=1, regularizer=None):
    layer = tf.keras.layers.Conv2D(n_filters, kernel, padding=padding, strides=strides, name=name, kernel_regularizer=regularizer)(input_layer)
    layer = tfa.layers.InstanceNormalization(axis=1)(layer) 
    return tf.keras.layers.LeakyReLU()(layer)

def isensee2017_classification_2d(input_shape,
                                nb_classes = 2,
                                n_base_filters=16,
                                context_dropout_rate=0.3,
                                gap_dropout_rate=0.4,                                                           
                                regularizer=None):
#     K.clear_session()
    inputs = tf.keras.Input(input_shape)
    depth = 5
    filters = [(2 ** i) * n_base_filters for i in range(depth)]
    # level 1: input --> conv_1 (stride = 1) --> context_1 --> summation_1
    conv_1 = create_convolution_block(inputs, filters[0], regularizer=regularizer)
    context_1 = create_context_module(conv_1, filters[0], dropout_rate=context_dropout_rate, regularizer=regularizer)
    summation_1 = tf.keras.layers.Add()([conv_1, context_1])
    # level 2: summation_1 --> conv_2 (stride = 2) --> context_2 --> summation_2
    conv_2 = create_convolution_block(summation_1, filters[1], strides=2, regularizer=regularizer)
    context_2 = create_context_module(conv_2, filters[1], dropout_rate=context_dropout_rate, regularizer=regularizer)
    summation_2 = tf.keras.layers.Add()([conv_2, context_2])
    # level 3: summation_2 --> conv_3 (stride = 2) --> context_3 --> summation_3
    conv_3 = create_convolution_block(summation_2, filters[2], strides=2, regularizer=regularizer)
    context_3 = create_context_module(conv_3, filters[2], dropout_rate=context_dropout_rate, regularizer=regularizer)
    summation_3 = tf.keras.layers.Add()([conv_3, context_3])
    # level 4: summation_3 --> conv_4 (stride = 2) --> context_4 --> summation_4
    conv_4 = create_convolution_block(summation_3, filters[3], strides=2, regularizer=regularizer)
    context_4 = create_context_module(conv_4, filters[3], dropout_rate=context_dropout_rate, regularizer=regularizer)
    summation_4 = tf.keras.layers.Add()([conv_4, context_4])
    # level 5: summation_4 --> conv_5 (stride = 2) --> context_5 --> summation_5
    conv_5 = create_convolution_block(summation_4, filters[4], strides=2, regularizer=regularizer)
    context_5 = create_context_module(conv_5, filters[4], dropout_rate=context_dropout_rate, regularizer=regularizer)
    summation_5 = tf.keras.layers.Add()([conv_5, context_5])
    clsfctn_op_GAP_summation_5 = tf.keras.layers.GlobalAveragePooling2D()(summation_5)
    if gap_dropout_rate:
        aggregated_maps = tf.keras.layers.Dropout(rate=gap_dropout_rate)(clsfctn_op_GAP_summation_5)
    clsfctn_Dense = tf.keras.layers.Dense(nb_classes, name="Dense_without_softmax", kernel_regularizer=regularizer)(aggregated_maps)
    clsfctn_op = tf.keras.layers.Activation('softmax', name="clsfctn_op")(clsfctn_Dense)
    model_clsfctn = tf.keras.Model(inputs=inputs, outputs=clsfctn_op)
    model_clsfctn.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), loss='binary_crossentropy', metrics=['acc',f1_m,precision_m])
    return model_clsfctn

## -------- END Model architecture --------- ##



## -Train function --##
#this function trains the neural network based on user defined arguments
def train_neural_network(name = "", save = True, epochs=27, batch = 20, px_size=50, train=False, tstSize = 0.3, randstate = 50, underSmp = False):
    model = None

    #if model name provided then it will load that specific model NOTE model must have been created with same pixel input size as dataset size otherwise an error will occur (I have not played with models outside of the ones generated by this code so that warning may not apply to you depending on the loaded model's architecture)
    if name is not None: 
        print("loading model..",name)
        model = load(name)
    else: 
        #model is built with user defined pixel sizes, please NOTE that this pixel size must be the samesize as the the data set pixel size 
        model = isensee2017_classification_2d([arguments.pxl_size,arguments.pxl_size,1],
                                    nb_classes = 2,
                                    n_base_filters=16,
                                    context_dropout_rate=0.4,
                                    gap_dropout_rate=0.4,                                                        
                                    regularizer=None)  
    image_train, image_test, image_val, label_train, label_test, label_val, uids_train, uids_test, uids_val = splitData(px_size, False, testSize = tstSize, randState = randstate, underSamp = underSmp)
    xt = tf.stack(image_train)
    yt = tf.stack(label_train)
    xv = tf.stack(image_test)
    yv = tf.stack(label_test)
    callbkarry = []

    #if train is true model will be trained using the train data set and correspondigly validated 
    if train:
        if arguments.val_loss_monitor is not None:
            val_loss_monitor = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=arguments.val_loss_monitor)
            callbkarry.append(val_loss_monitor)
        if arguments.loss_monitor is not None:
            loss_monitor = tf.keras.callbacks.EarlyStopping(monitor='loss', restore_best_weights=True, patience=arguments.loss_monitor)
            callbkarry.append(loss_monitor)
        history = model.fit(
            xt,
            yt,
            batch_size= batch,
            epochs=epochs,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=(xv, yv),
            callbacks=callbkarry,
        )

    #if you choose to save the model it will be saved in the saveLocation folder 
    if save: 
        now = datetime.datetime.now()
        saveModel(model, str(now.time()))
    
    #finally model is tested using test datasets and results are printed on screen 
    label_test = tf.stack(label_test)
    image_test = tf.stack(image_test)
    testing_results = model.evaluate(image_test, label_test, batch_size=batch)
    return model, image_train, image_test, image_val, label_train, label_test, label_val, uids_train, uids_test, uids_val, testing_results


model, image_train, image_test, image_val, label_train, label_test, label_val, uids_train, uids_test, uids_val, testing_results = train_neural_network(name = arguments.model_name, save = arguments.save, epochs = arguments.epochs, batch = arguments.batch_size, px_size = arguments.pxl_size, train=arguments.train, tstSize = arguments.test_size, randstate = arguments.rand_state, underSmp = arguments.underSmp)

#data set stats are printed for user convenience 
print("total subjects used in training", len(uids_train))
print("total subjects used in validaion", len(uids_val))
print("total subjects used in testing", len(uids_test))

#results of testing a printed as well
print("testing results [loss, acc, f1_m, precision_m] \n", testing_results)
