
import os
import numpy as np
from keras.models import Sequential # To initialise the nn as a sequence of layers
from keras.layers import Convolution2D # To make the convolution layer for 2D images
from keras.layers import MaxPooling2D # 
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import CSVLogger
from keras.optimizers import RMSprop
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

csv=CSVLogger("7_Encoding_with_activation.log")
#filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32,(2,2),input_shape = (224,224,1),strides=2,name='convo1'))
classifier.add(Convolution2D(64,(3,3), activation = 'relu',name='convo2'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 1 - Convolution
classifier.add(Convolution2D(64,(3,3),activation = 'relu',name='convo3'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(BatchNormalization())
classifier.add(GlobalAveragePooling2D())
# As our model is still facing the problem so, we need to increase the regulization
classifier.add(Dropout((0.5)))
classifier.add(Dense(20, activation = 'softmax'))
Using TensorFlow backend.
classifier.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

curr_path = os.getcwd()
basefolder = os.path.dirname(curr_path)

train_folder = os.path.join(basefolder, "Dataset\\Train")
test_folder = os.path.join(basefolder, "Dataset\\dev")
"""
# Changes for linux
train_folder = os.path.join(basefolder, "Dataset/Train")
test_folder = os.path.join(basefolder, "Dataset/dev")
"""
train_set = train_datagen.flow_from_directory(train_folder,target_size=(224, 224),batch_size=64,class_mode='categorical',color_mode='grayscale')

test_set = test_datagen.flow_from_directory(test_folder,target_size=(224, 224),batch_size=64,class_mode='categorical',color_mode='grayscale')
Found 2600 images belonging to 20 classes.
Found 200 images belonging to 20 classes.
history = classifier.fit_generator(train_set,steps_per_epoch=2600,epochs=10,validation_data=test_set,validation_steps=200,callbacks=[csv],verbose=2)
Epoch 1/10
587s - loss: 2.6827 - acc: 0.1726 - val_loss: 3.0515 - val_acc: 0.1350
Epoch 2/10
489s - loss: 2.0286 - acc: 0.3802 - val_loss: 2.7357 - val_acc: 0.1298
Epoch 3/10
492s - loss: 1.4900 - acc: 0.5453 - val_loss: 2.8891 - val_acc: 0.1742
Epoch 4/10
489s - loss: 1.1193 - acc: 0.6578 - val_loss: 2.9546 - val_acc: 0.1602
Epoch 5/10
493s - loss: 0.8672 - acc: 0.7333 - val_loss: 2.7079 - val_acc: 0.2997
Epoch 6/10
493s - loss: 0.6843 - acc: 0.7892 - val_loss: 3.1823 - val_acc: 0.2748
Epoch 7/10
596s - loss: 0.5537 - acc: 0.8284 - val_loss: 3.3980 - val_acc: 0.2950
Epoch 8/10
797s - loss: 0.4605 - acc: 0.8572 - val_loss: 3.1704 - val_acc: 0.3400
Epoch 9/10
730s - loss: 0.3899 - acc: 0.8789 - val_loss: 4.0978 - val_acc: 0.2945
Epoch 10/10
505s - loss: 0.3334 - acc: 0.8959 - val_loss: 3.5963 - val_acc: 0.2799
classifier.save("7_Encoding_with_activation.h5")