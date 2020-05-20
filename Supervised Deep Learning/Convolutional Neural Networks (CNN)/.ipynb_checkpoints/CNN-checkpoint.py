#!/usr/bin/env python
# coding: utf-8

# ### Image classification using convolutional nueral networks 

# In[1]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


# intializing CNN
imageClassifier = Sequential()

# step 1 of CNN is adding convolution layer - convolving our images with feature extracter matrix 
imageClassifier.add(Conv2D(32,3,3,input_shape = (64,64,3),activation = 'relu'))
imageClassifier.add(Conv2D(32,3,3,input_shape = (64,64,3),activation = 'relu'))

# step 2 of CNN is adding a pooling layer
imageClassifier.add(MaxPooling2D(pool_size = (2,2),strides = 2))
imageClassifier.add(Dropout(0.2))

imageClassifier.add(Conv2D(32,3,3,input_shape = (64,64,3),activation = 'relu'))
imageClassifier.add(Conv2D(32,3,3,input_shape = (64,64,3),activation = 'relu'))
imageClassifier.add(MaxPooling2D(pool_size = (2,2),strides = 2))
imageClassifier.add(Dropout(0.2))



# step 3 of a CNN is adding a flattening layer
imageClassifier.add(Flatten())

# step 4 in CNN is to add hidden layers
imageClassifier.add(Dense(units = 128,activation = 'relu'))
imageClassifier.add(Dropout(0.25))
imageClassifier.add(Dense(units = 128,activation = 'relu'))
imageClassifier.add(Dropout(0.25))
imageClassifier.add(Dense(units = 128,activation = 'relu'))
imageClassifier.add(Dropout(0.25))
imageClassifier.add(Dense(units = 1 ,activation = 'sigmoid'))

#step 5 in CNN is to compile our CNN layers
imageClassifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])


# step 6 in CNN is to fit the model on images
""" 
we do image aggumentation to reduce overfitting by enriching our dataset without adding more data images,
rather by looking at the available images differently everytime 
 we use ImageDataGenerator for this oricess
"""
train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                                                    'dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
                                                    'dataset/test_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')
imageClassifier.fit(
                    train_generator,
                    steps_per_epoch=8000,
                    epochs=1,
                    validation_data=validation_generator,
                    validation_steps=2000)
    
    


# In[ ]:


# look like our modeling is overfitting


# In[ ]:




