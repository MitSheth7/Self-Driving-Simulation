from utils import *
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

print('Setting UP')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# Step 3: Prepare for processing
path = 'myData'
data = importDataInfo(path)
data = balanceData(data, display=False)

# Prepare data for processing
imagesPath, steerings = loadData(path, data)

# Step 4: Split for Training and Validation
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=10)

print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

# Step 7: Batch Generator Example (for training)
batch_size = 32
train_generator = batchGen(xTrain, yTrain, batch_size, trainFlag=True)
val_generator = batchGen(xVal, yVal, batch_size, trainFlag=False)

# Function to create the model
def createModel():
    model = Sequential()
    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))  # Output layer for steering angle

    model.compile(Adam(lr=0.0001), loss='mse')  # Compile the model with Adam optimizer and MSE loss
    return model

# Create and summarize the model
model = createModel()
model.summary()

# Step 9: Training
history = model.fit(train_generator,
                    steps_per_epoch=300,
                    epochs=10,
                    validation_data=val_generator,
                    validation_steps=200)

# Step 10: Saving & Plotting
model.save('model.h5')
print('Model Saved')

# Plotting training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
