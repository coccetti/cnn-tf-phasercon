#!/usr/bin/env python3
# CNN on phase images
# 17/04/2024

# Inputs are npy-files: measured_phase.py
# with dim: 1200x 1920
# and min: -PI and max: +PI
# xcopy .\*_phase.npy C:\Users\LFC_01\destination\ /S /C /Y

#%% Imports
import tensorflow as tf
print("\nTensorflow and Keras versions:")
print(tf.__version__)
print(tf.keras.__version__)
from keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2


#%% Parameters

# Number of RUN
nRUN=40

# Data path
# data_path = r'C:\Users\LFC_01\Desktop\Szilard\Data'
data_path = r'data2'

# Date of measurements
# date='2024_04_04'
date=''
    
# Define the type of measure, so we can read the proper folder
# data_type_measure = "SLM_Interferometer_alignment"
data_type_measure = ''


# Classes
classes=["input_mask_blank_screen", 'input_mask_vertical_division', 'input_mask_horizontal_division', 'input_mask_checkboard_1', 'input_mask_checkboard_2', 'input_mask_checkboard_3', 'input_mask_checkboard_4']
classes_short=np.arange(np.size(classes))
print("\nClasses in words:", classes)
print("\nClasses in numbers: ", classes_short)

# Measured Phase file names
measured_phase_input_file = "measured_phase.npy"

# Number of frames
Nframes=nRUN*np.size(classes)
print("\nNframes (nRUN * classes): ", Nframes)

#%% Determine high and widht of the matrix reading the data

sample_measured_phase_full_path = os.path.join(data_path, date, data_type_measure, "M00001", classes[0], "files", measured_phase_input_file)
sample_measured_phase = np.load(sample_measured_phase_full_path)
high, width = sample_measured_phase.shape
print("\nLoaded sample phase with dimensions (high, width): ", high, width)


#%% Load data and plot

# initialization of the frame matrix
measured_phase = np.zeros((Nframes, high, width))
reference_class_short = np.zeros((Nframes, 1), dtype=int)

j=0 #  index for NFrames (varies from 0 to Nframes-1)
for i in range(nRUN):
    RUN = 'M0000' + str(i+1)
    if i+1 > 9:
        RUN = 'M000' + str(i+1)
        if i+1 > 99:
            RUN = 'M00' + str(i+1)
    k=0  # index for classes (varies from 0 to )
    for class_name in classes :
        input_files_path = os.path.join(data_path, date,data_type_measure, RUN, class_name, "files", measured_phase_input_file)
        measured_phase[j,:,:] = np.load(input_files_path)
        reference_class_short[j] = k
        # ### Plot figure
        # img=plt.imshow(measured_phase[:,:,j])#cmap='Blues'
        # plt.axis('off')
        # plt.colorbar(img)
        # plt.show()
        # plt.pause(0.1)
        # print(input_files_path)
        # print(j)
        
        # ### Test print values of the measured phase
        # print("\nShape of measured phase: ", measured_phase.shape)
        # print(reference_class_short.shape)
        # print('min of measured_phase: ', measured_phase.min())
        # print('max of measured_phase: ', measured_phase.max())

        k+=1
        j+=1

# Print arrays dimensions
print("- measured_phase tensor shape: ", tf.shape(measured_phase).numpy())
print("- reference_class_short tensor shape: ", reference_class_short.shape)


#%% Resize images
x_crop_size = 128
y_crop_size = 128
print(measured_phase.shape)
resize_measured_phase = np.zeros((Nframes, x_crop_size, y_crop_size))

for ii in range(Nframes):
    resize_measured_phase[ii] = cv2.resize(measured_phase[ii], (x_crop_size, y_crop_size), interpolation = cv2.INTER_NEAREST)

# reassign values just to make things works simplier
measured_phase = resize_measured_phase

#%% Crop images
# n_crops_per_image = 10  # number of crops per image
# x_crop_size = 64  # number of pixels for x axis
# y_crop_size = 64  # number of pixels for x axis
# tf.random.set_seed(1234)

# # crop_measured_phase = tf.image.random_crop(value=measured_phase, size=(Nframes, x_crop_size, y_crop_size))
# crop_measured_phase = tf.image.stateless_random_crop(value=measured_phase, size=(Nframes, x_crop_size, y_crop_size), seed=(1,0))
# crop_reference_class_short = reference_class_short
# for ii in range(n_crops_per_image-1):
#     # crop_measured_phase_temp = tf.image.random_crop(value=measured_phase, size=(Nframes, x_crop_size, y_crop_size))
#     crop_measured_phase_temp = tf.image.stateless_random_crop(value=measured_phase, size=(Nframes, x_crop_size, y_crop_size), seed=(1,0))
#     crop_measured_phase = np.concatenate((crop_measured_phase, crop_measured_phase_temp))
#     crop_reference_class_short = np.concatenate((crop_reference_class_short, reference_class_short))

# print("crop_measured_phase: ", tf.shape(crop_measured_phase).numpy())
# print("crop_reference_class_short: ", crop_reference_class_short.shape)

# # reassign values just to make things works simplier
# measured_phase = crop_measured_phase
# reference_class_short = crop_reference_class_short
# Nframes = Nframes*n_crops_per_image


#%% Split data between Train and Test, then Normalization
train_set_percentage=0.9 # Set training set percentage
print(np.int32(Nframes * train_set_percentage))
print(Nframes-np.int32(Nframes*train_set_percentage))

train_measured_phase, test_measured_phase = tf.split(measured_phase, [np.int32(Nframes * train_set_percentage),Nframes - np.int32(Nframes * train_set_percentage)])
train_reference_class_short, test_reference_class_short = tf.split(reference_class_short, [np.int32(Nframes * train_set_percentage),Nframes-np.int32(Nframes*train_set_percentage)])
print("- train_measured_phase tensor shape: ", tf.shape(train_measured_phase).numpy())
print("- test_measured_phase tensor shape: ", tf.shape(test_measured_phase).numpy())
print("- train_reference_class_short tensor shape: ", tf.shape(train_reference_class_short).numpy())
print("- test_reference_class_short tensor shape: ", tf.shape(test_reference_class_short).numpy())

# Normalize "pixel" values to be between 0 and 1
train_measured_phase = (train_measured_phase - np.min(train_measured_phase)) / (np.max(train_measured_phase)-np.min(train_measured_phase))
test_measured_phase = (test_measured_phase - np.min(test_measured_phase)) / (np.max(test_measured_phase)-np.min(test_measured_phase))

# # Verify the data
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_measured_phase[i])
#     plt.xlabel(int(train_reference_class_short[i]))
# plt.show()
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_measured_phase[i+180])
#     plt.xlabel(int(train_reference_class_short[i+180]))
# plt.show()
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(test_measured_phase[i])
#     plt.xlabel(int(test_reference_class_short[i]))
# plt.show()
# sys.exit()

#%% Create the convolutional base
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(x_crop_size, y_crop_size,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Display the model
model.summary()

#%% Add Dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(7)) # TODO change 7 in number of classes 
# Display the model
model.summary()

#%% Compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_measured_phase, train_reference_class_short, epochs=10, 
                    validation_data=(test_measured_phase, test_reference_class_short))


#%% Evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_measured_phase, test_reference_class_short, verbose=2)
print("test acc: ", test_acc)


sys.exit()
