import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix
import ipywidgets as widgets
import io
from PIL import Image
from numpy import expand_dims
#from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from IPython.display import display,clear_output
from warnings import filterwarnings
from keras.models import Sequential,Model
from keras.layers import MaxPooling2D,Conv2D,Dense,BatchNormalization,Dropout,GlobalAveragePooling2D,Flatten,Input
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Create your views here.
def classification(request):
    colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
    colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
    colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']

    sns.palplot(colors_dark)
    sns.palplot(colors_green)
    sns.palplot(colors_red)
    labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']
    X_train = []
    y_train = []
    image_size = 150
    for i in labels:
        folderPath = os.path.join('D:/PBS/labelled Dataset','Training',i)
        for j in tqdm(os.listdir(folderPath)):
            img = cv2.imread(os.path.join(folderPath,j))
            img = cv2.resize(img,(image_size, image_size))
        
       #selecting only no tumor images sorting by labels 
            if i=='no_tumor':
                data = img_to_array(img)
            #expand ranks of the images
                samples = expand_dims(data, 0)
            
            # create image data augmentation generator
            
                datagen = ImageDataGenerator(rotation_range=90 , horizontal_flip=True)
            
            
            # prepare iterator
            
                it = datagen.flow(samples, batch_size=1)
            
           #generate samples and plot
                for k in range(2):
                
                # define subplot
                #pyplot.subplot(330 + 1 + i)
                
                
                # generate batch of images
                    batch = it.next()
               
                
                # convert to unsigned integers for viewing
                    image = batch[0].astype('uint8')
                
                #pyplot.imshow(image)
                
            # show the figure
            #pyplot.show()
                
                    X_train.append(image)
                    y_train.append(i)
            else:
                X_train.append(img)
                y_train.append(i)
        
    for i in labels:
        folderPath = os.path.join('D:/PBS/labelled Dataset','Testing',i)
        for j in tqdm(os.listdir(folderPath)):
            img = cv2.imread(os.path.join(folderPath,j))
            img = cv2.resize(img,(image_size,image_size))
            X_train.append(img)
            y_train.append(i)
        
    X_train = np.array(X_train)
#print X_train lenght to check if the array length increase by 395
    print(len(X_train))
    y_train = np.array(y_train)
    k=0
    fig, ax = plt.subplots(1,4,figsize=(20,20))
    fig.text(s='Sample Image From Each Label',size=18,fontweight='bold',
             fontname='monospace',color=colors_dark[1],y=0.62,x=0.4,alpha=0.8)
    for i in labels:
        j=0
        while True :
            if y_train[j]==i:
                ax[k].imshow(X_train[j])
                ax[k].set_title(y_train[j])
                ax[k].axis('off')
                k+=1
                break
            j+=1
    X_train, y_train = shuffle(X_train,y_train, random_state=101)
    X_train.shape
    X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size=0.1,random_state=101)
    y_train_new = []
    for i in y_train:
        y_train_new.append(labels.index(i))
    y_train = y_train_new
    y_train = tf.keras.utils.to_categorical(y_train)


    y_test_new = []
    for i in y_test:
        y_test_new.append(labels.index(i))
    y_test = y_test_new
    y_test = tf.keras.utils.to_categorical(y_test)
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(150, 150, 3), padding = 'Same'))
    model.add(Conv2D(32, kernel_size=(3, 3),  activation ='relu', padding = 'Same'))


    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size = (3,3), activation ='relu', padding = 'Same'))
#model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))

#model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size = (3,3), activation ='relu', padding = 'Same'))
#model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))

#model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss = "categorical_crossentropy", optimizer='Adam')
    print(model.summary())
    print(model.output)
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer = 'Adam', metrics= ['accuracy'])
    tensorboard = TensorBoard(log_dir = 'logs')
#checkpoint = ModelCheckpoint("vgg16.h5",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,
                              mode='auto',verbose=1)
    history=model.fit(X_train,y_train,validation_split=0.1, epochs =20, verbose=1, batch_size=32,
                   validation_data=(X_test, y_test),callbacks=[tensorboard,reduce_lr])
    y_pred= model.predict(X_test)
    y_pred = np.argmax(y_pred,axis=1)
    y_pred[:15] 
    y_test_new = np.argmax(y_test,axis=1)     
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)  
    y_train = to_categorical(y_train)#num_classes = 4
    y_train.shape
    y_test = to_categorical(y_test)#num_classes = 4
    y_test.shape       
    plt.figure(figsize=(18,12))
    for i in range(70):
        pred_res = "Correctly predicted!"
        sample_idx = random.choice(range(len(X_test)))
        plt.subplot(7,10,i+1)
        plt.imshow(X_test[sample_idx])
        if y_pred[sample_idx] != y_test_new[sample_idx]:
            pred_res = "Mispredicted!"
        plt.xlabel(f"Actual: {y_test_new[sample_idx]}\n Predicted: {y_pred[sample_idx]}\n {pred_res}")
    
    plt.tight_layout()
    plt.show()