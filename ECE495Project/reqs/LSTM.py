#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install tensorflow')


# In[2]:


get_ipython().system('pip3 install opencv-python')
import tensorflow as  tf
from tensorflow.keras.models import Sequential, Model, model_from_json, load_model
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import string
import tensorflow.keras as keras
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import load_img, img_to_array




# In[230]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[231]:


import IPython
print(IPython.sys_info())


# In[232]:


TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
mnist = tf.keras.datasets.mnist

# train is now 60% of the entire data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

X = numpy.concatenate([x_train, x_test])
y = numpy.concatenate([y_train, y_test])

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=(1-TRAIN_RATIO))
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=((TEST_RATIO/(VALIDATION_RATIO+TEST_RATIO))))
#Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis = -1)
x_test = tf.keras.utils.normalize(x_test, axis = -1)
x_val = tf.keras.utils.normalize(x_val, axis = -1)


# # Plotting some of the data

# First 16 images:

# In[233]:


fig, axs = plt.subplots(4, 4)
count = 0
for i in range(4):
    for j in range(4):
        axs[i,j].imshow(x_train[count])
        count += 1


# # Basic ANN Model

# In[234]:


baseModel = Sequential()
baseModel.add(Flatten(input_shape=(28,28)))
baseModel.add(Dense(units = 128, activation = 'relu'))
baseModel.add(Dense(units = 128, activation = 'relu'))
baseModel.add(Dense(units = 20, activation = 'softmax'))
baseModel.summary()

baseModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
baseModel.fit(x_train, y_train, validation_data = (x_val, y_val), epochs=10, batch_size=100)


# In[235]:


loss, accuracy = baseModel.evaluate(x_test, y_test)
print("Loss : ", loss)
print("Accuracy : ", accuracy)


# In[236]:



y_pred = baseModel.predict(x_test, batch_size=64, verbose=1)
y_pred_bool = numpy.argmax(y_pred, axis=1)



print(classification_report(y_test, y_pred_bool))


prediction = baseModel.predict([x_test])
print('Prediction: ', numpy.argmax(prediction[20]))
plt.imshow(x_test[20])
plt.show()


# In[237]:


baseModel_json = baseModel.to_json()
with open("baseModel.json", "w") as json_file:
    json_file.write(baseModel_json)
baseModel.save_weights("model.h5")

baseModeljsonfile = open('baseModel.json', 'r')
loaded_baseModel_json = baseModeljsonfile.read()
baseModeljsonfile.close()
baseModelNew = model_from_json(loaded_baseModel_json)
baseModelNew.load_weights("model.h5")

baseModelNew.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
loss, accuracy = baseModelNew.evaluate(x_test, y_test)

print("Loss : ", loss)
print("Accuracy : ", accuracy)


# In[238]:


def load_image(filename):
    img = load_img(filename, grayscale = True, target_size=(28,28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)

    img = img.astype('float32')
    img = img/255.0
    return img


# In[242]:


file = input("What is the file name")
img = load_image(file)
predict_value = baseModelNew.predict(img)
digit = argmax(predict_value)
print(digit)


# In[243]:


loss1 = str(loss)
accuracy1 = str(accuracy)
digit1 = str(digit)


# In[244]:


with open('resultsANN.txt', 'w') as f:
    f.write("This is the loss \n")
    f.writelines(loss1)
    f.write("\n")
    f.write("This is the accuracy \n")
    f.writelines(accuracy1)
    f.write("\n")
    f.write("Based on the file inputted, the ANN Model predicts that the value is :")
    f.write("\n")
    f.writelines(digit1)
    f.close()


# # Bi-Directional LSTM Implementation
# 

# In[245]:


from tensorflow.keras.layers import Bidirectional


# In[246]:


bmodel = Sequential([
    Bidirectional(LSTM(256, input_shape=(28,28), return_sequences=True, activation='relu')),
    Bidirectional(LSTM(256, activation='relu')),
    Dense(10, activation='softmax')  
    ])
bmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early = EarlyStopping(patience=5)
bmodel.fit(x_train, y_train, validation_data = (x_val, y_val), verbose=1, batch_size=64, epochs=5, callbacks=early)
bmodel.evaluate(x_test, y_test)


# In[170]:


loss, accuracy = bmodel.evaluate(x_test, y_test)
print("loss: ", loss)
print("accuracy: ", accuracy)

y_pred = bmodel.predict(x_test, batch_size=64, verbose=1)
y_pred_bool = numpy.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_bool))

prediction = bmodel.predict([x_test])
print('Prediction: ', numpy.argmax(prediction[13]))
plt.imshow(x_test[13])
plt.show()


# In[191]:


bdLSTMModel_json = bmodel.to_json()
with open("baseModel.json", "w") as json_file:
    json_file.write(bdLSTMModel_json)
bmodel.save_weights("bdModel.h5")

bdLSTMModeljsonfile = open('baseModel.json', 'r')
loaded_bdLSTMModel_json = bdLSTMModeljsonfile.read()
bdLSTMModeljsonfile.close()
bdLSTMModelNew = model_from_json(loaded_bdLSTMModel_json)
bdLSTMModelNew.load_weights("bdModel.h5")

bdLSTMModelNew.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
loss, accuracy = bdLSTMModelNew.evaluate(x_test, y_test)


# In[200]:


loss1 = str(loss)
accuracy1 = str(accuracy)

    


# In[201]:


file = input("What is the file name")
img = load_image(file)
predict_value = bdLSTMModelNew.predict(img)
digit = argmax(predict_value)
print(digit)
digit1 = str(digit)


# In[206]:


with open('resultsBiDirectionalLSTM.txt', 'w') as f:
    f.write("This is the loss \n")
    f.writelines(loss1)
    f.write("\n")
    f.write("This is the accuracy \n")
    f.writelines(accuracy1)
    f.write("\n")
    f.write("Based on the file inputted, the Bi Directional LSTM Model predicts that the value is :")
    f.write("\n")
    f.writelines(digit1)
    f.close()


# # RNN-LSTM Implementation

# In[113]:


es_callback = EarlyStopping(patience = 2)

model = Sequential()
#CuDNNLSTM

model.add(LSTM(256, input_shape=(x_train.shape[1:]),  return_sequences=True, activation='relu'))
model.add(LSTM(256))
model.add(Dense(64, activation='softmax'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.summary()
          

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose = 1, batch_size=128, epochs=2, callbacks=[es_callback])
results = model.evaluate(x_test, y_test)
print("test loss, test acc:", results)

model.summary()
loss, accuracy = model.evaluate(x_test, y_test)
print("Loss : " , loss)
print("Accuracy : " , accuracy)

y_pred = model.predict(x_test, batch_size=64, verbose=1)
y_pred_bool = numpy.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_bool))

prediction = model.predict([x_test])
print('Prediction: ', numpy.argmax(prediction[10]))
plt.imshow(x_test[10])
plt.show()


# In[114]:


es_callback = EarlyStopping(patience = 3)
model4 = Sequential()
model4.add(LSTM(128,input_shape=(x_train.shape[1:]),  return_sequences=True, activation='relu') )
model4.add(LSTM(128))
model4.add(Dropout(0.2))
model4.add(Dense(10, activation='relu'))
model4.summary()


          

model4.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model4.fit(x_train, y_train, validation_data=(x_val, y_val), verbose = 1, batch_size=128, epochs=5, callbacks=[es_callback])
results = model4.evaluate(x_test, y_test)
print("test loss, test acc:", results)

model4.summary()
loss, accuracy = model4.evaluate(x_test, y_test)
print("Loss : " , loss)
print("Accuracy : " , accuracy)

y_pred = model4.predict(x_test, batch_size=64, verbose=1)
y_pred_bool = numpy.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_bool))

prediction = model4.predict([x_test])
print('Prediction: ', numpy.argmax(prediction[10]))
plt.imshow(x_test[10])
plt.show()


# In[115]:


callbacksbb = EarlyStopping(patience = 2)
model3 = Sequential()
model3.add(LSTM(64,input_shape=(x_train.shape[1:]),  return_sequences=True,  activation ='relu') )
model3.add(LSTM(64))
model3.add(Dropout(0.1))
model3.add(Dense(32, activation='softmax'))
model3.add(Dropout(0.1))
model3.add(Dense(10, activation='softmax'))
       
model3.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model3.fit(x_train, y_train, validation_data=(x_val, y_val), verbose = 1, batch_size=128, epochs=10, callbacks=[callbacksbb])
results = model3.evaluate(x_test, y_test)
print("test loss, test acc:", results)

model3.summary()
loss, accuracy = model3.evaluate(x_test, y_test)
print("Loss : " , loss)
print("Accuracy : " , accuracy)

y_pred = model3.predict(x_test, batch_size=64, verbose=1)
y_pred_bool = numpy.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_bool))

prediction = model3.predict([x_test])
print('Prediction: ', numpy.argmax(prediction[10]))
plt.imshow(x_test[10])
plt.show()


# # CNN Implementation

# Reshaping the Data

# In[215]:


x_train1 = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val1 = x_val.reshape(x_val.shape[0], 28, 28 ,1)
x_test1 = x_test.reshape(x_test.shape[0], 28, 28, 1)


# In[218]:


# Model
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), strides=2,padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), strides=2,padding='same', activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train1,y=y_train, validation_data=(x_val1, y_val), epochs=4)


# In[219]:


model.summary()
loss, accuracy = model.evaluate(x_test1, y_test)
print("Loss : " , loss)
print("Accuracy : " , accuracy)

y_pred = model.predict(x_test1, batch_size=64, verbose=1)
y_pred_bool = numpy.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_bool))

prediction = model.predict([x_test1])
print('Prediction: ', numpy.argmax(prediction[10]))
plt.imshow(x_test1[10])
plt.show()


# In[220]:


def load_image(filename):
    img = load_img(filename, grayscale = True, target_size=(28,28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)

    img = img.astype('float32')
    img = img/255.0
    return img
file = input("What is the file name")
img = load_image(file)
predict_value = baseModelNew.predict(img)
digit = argmax(predict_value)
print(digit)
loss1 = str(loss)
accuracy1 = str(accuracy)
digit1 = str(digit)
with open('resultsCNN.txt', 'w') as f:
    f.write("This is the loss \n")
    f.writelines(loss1)
    f.write("\n")
    f.write("This is the accuracy \n")
    f.writelines(accuracy1)
    f.write("\n")
    f.write("Based on the file inputted, the CNN Model predicts that the value is :")
    f.write("\n")
    f.writelines(digit1)
    f.close()


# # Training Different CNN Architectures

# # GRU Implementation

# In[221]:


from tensorflow.keras.layers import GRU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# In[223]:


gru = Sequential([
    GRU(256, input_shape=(28,28), return_sequences=True, activation='relu'),
    GRU(256, activation='relu'),
    Dense(10, activation='softmax')  
    ])
gru.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
gru.summary()
# Callbacks
gru.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=1, batch_size=64, epochs=4)




# In[224]:


loss, accuracy = gru.evaluate(x_test, y_test)
print("loss : " , loss)
print("accuracy : " , accuracy)


# In[225]:


y_pred = gru.predict(x_test, batch_size=64, verbose=1)
y_pred_bool = numpy.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_bool))


# In[226]:


prediction = gru.predict([x_test])
print('Prediction: ', numpy.argmax(prediction[15]))
plt.imshow(x_test[15])
plt.show()


# In[227]:


def load_image(filename):
    img = load_img(filename, grayscale = True, target_size=(28,28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)

    img = img.astype('float32')
    img = img/255.0
    return img
file = input("What is the file name")
img = load_image(file)
predict_value = baseModelNew.predict(img)
digit = argmax(predict_value)
print(digit)
loss1 = str(loss)
accuracy1 = str(accuracy)
digit1 = str(digit)
with open('resultsGRU.txt', 'w') as f:
    f.write("This is the loss \n")
    f.writelines(loss1)
    f.write("\n")
    f.write("This is the accuracy \n")
    f.writelines(accuracy1)
    f.write("\n")
    f.write("Based on the file inputted, the GRU Model predicts that the value is :")
    f.write("\n")
    f.writelines(digit1)
    f.close()


# In[ ]:


get_ipython().run_cell_magic('writefile', 'requirements.txt', '')

