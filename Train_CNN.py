# step 2 - Training the deep learning model and Storing the trained deep learning model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np

# Read the Data frame
preprocessed_data = pd.read_csv('train.csv', index_col=False)
# fetches 784th column
preprocessed_data.head()
labels = preprocessed_data[['784']]
# Dropping the target variable from the dataframe to get only the features
preprocessed_data.drop(preprocessed_data.columns[[784]], axis=1, inplace=True)

# Convert labels series to numpy array
labels = np.array(labels)
# print(labels)

# Use the keras to_categorical function to apply one hot encoding
cat = to_categorical(labels, num_classes=14)
# print(cat.shape)

final = []
# Iterate over the number of rows in the data
for i in range(61859):
    # Reshape to 28x28 and append to a list
    final.append(np.array(preprocessed_data[i:i+1]).reshape(28, 28, 1))


model = Sequential()
# in this step the 28*28 image will be multiplied by the kernel of size 5*5
model.add(Conv2D(16, kernel_size=(5, 5), input_shape=(28, 28, 1), data_format='channels_last', activation='relu'))
# Pooling is the process of merging. So it’s basically for the purpose of reducing the size of the data.
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the final layers or fully connected layers are for classification -- used for changing the dimensions of the vector.
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(14, activation='softmax'))
#Softmax -- converts ouput values to probability -- e^out / wum e^allpossout
model.summary()
# Compile model
# categorical crossentropy to learn to give a high probability to the correct digit and a low probability to the other digits.
#Softmax -- activation function recommended to use
# sumission - tilog(yi), where t = target, y = output
# metrics - used to judge the performance of your model -- Calculates how often predictions equal labels.
# optimizer -- algorithm to update -- parameters --  reduce loss
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
# purpose of this fit function is used to evaluate your model on training.
# epochs = no of times the model is needed to be evaluated during training
# One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
# with every epoch weight are changed in the neural network (for optimization)
# batch_size = total number of training examples present in a single batch.
model.fit(np.array(final), cat, epochs=10, batch_size=200, shuffle=True, verbose=1)

# trained model to json file
model_json = model.to_json()
with open("model_rev.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_rev.h5")
